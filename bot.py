### Script for one-shot Reddit bots using Huggingface models
## Unlike ssi-bot, these are *not* necessarily finetuned on data from any subreddit
## Rather, they are prompted with a "character" to play (name + backstory)
import requests
import praw
import csv
import random
import re
import time
import schedule
from datetime import datetime, date
import os, sys
from hf_utils import generate_text, query
from tagging_mixin import TaggingMixin
import yaml
import threading
from nltk import word_tokenize
from rake_nltk import Rake
from googleapiclient import discovery
import http.client, urllib.request, urllib.parse, urllib.error, base64
import json

_default_negative_keywords = [
    ('ar', 'yan'), ('ausch, witz'),
    ('black', ' people'),
    ('child p', 'orn'), ('concentrati', 'on camp'),
    ('fag', 'got'),
    ('hit', 'ler'), ('holo', 'caust'),
    ('inc', 'est'), ('israel'),
    ('jew', 'ish'), ('je', 'w'), ('je', 'ws'),
    ('k', 'ill'), ('kk', 'k'),
    ('lol', 'i'),
    ('maste', 'r race'), ('mus', 'lim'),
    ('nation', 'alist'), ('na', 'zi'), ('nig', 'ga'), ('nig', 'ger'),
    ('pae', 'do'), ('pale', 'stin'), ('ped', 'o'),
    ('rac' 'ist'), ('r', 'ape'), ('ra', 'ping'),
    ('sl', 'ut'), ('swas', 'tika'),
]

_negative_keywords = ["".join(s) for s in _default_negative_keywords]

## Load config details from YAML
def load_yaml(filename):
    with open(filename, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as error:
            print(error)
    return None

def words_below(string,max_words):
    # check to see if an input string would exceed token budget
    token_list = word_tokenize(string)
    if len(token_list)>max_words:
        return False
    else:
        return True

def clean_text(generated_text):
    truncate = 0
    cleanStr = ''
    # look for double-quotes
    truncate = generated_text.find('"')
    if truncate>-1:
        cleanStr = generated_text[:truncate]
    # if we can't find double-quotes, look for punctuation
    elif re.search(r'[?.!]', generated_text):
            trimPart = re.split(r'[?.!]', generated_text)[-1]
            cleanStr = generated_text.replace(trimPart,'')
    # if we can't find punctuation, use the last space
    else:
        truncate = generated_text.rfind(' ')
        if truncate>-1:
            cleanStr = generated_text[:truncate+1]
    if not cleanStr:
        print('Bad generation')
    return cleanStr

def get_keywords(text):
    rake_nltk_var = Rake()
    rake_nltk_var.extract_keywords_from_text(text)
    keyword_extracted = rake_nltk_var.get_ranked_phrases()[:10]
    return keyword_extracted

class reddit_bot:
    def __init__(self, config_file):
        self.config = load_yaml(config_file)
        if not self.config:
            print('Cannot load config file; check path and formatting')
            sys.exit()
        self.HF_key = os.environ[self.config['HF_key_var']]
        self.headers = {"Authorization": "Bearer "+self.HF_key}
        self.DeepAI_API_key = os.environ[self.config['deepai_api_key_var']]
        self.Google_API_key = os.environ[self.config['Google_API_key_var']]
        self.Azure_token = os.environ[self.config['azure_token_var']]
        # self.textsynth_key = os.environ[self.config['textsynth_key_var']]
        # self.config['reddit_pass'] = os.environ[self.config['reddit_pass_var']]
        # self.config['reddit_ID'] = os.environ[self.config['reddit_ID_var']]
        # self.config['reddit_secret'] = os.environ[self.config['reddit_secret_var']]
        self.reddit = praw.Reddit(
            user_agent=self.config['bot_username'],
            client_id=self.config['reddit_ID'],
            client_secret=self.config['reddit_secret'],
            username=self.config['bot_username'],
            password=self.config['reddit_pass'],
        )
        self.me = self.reddit.user.me()
        self.reddit.validate_on_submit = True
        self.sub = self.reddit.subreddit(self.config['bot_subreddit'])
        self.submission_writer = threading.Thread(target=self.submission_loop, args=())
        self.submission_reader = threading.Thread(target=self.watch_submissions, args=())
        self.inbox_reader = threading.Thread(target=self.watch_inbox, args=())
        self.today = date.today()
        self.tally = 0 # to compare with daily input character budget
        self.SSI = TaggingMixin() # handler for legacy SSI tagging functions
        self.negative_keywords = _negative_keywords + self.config['negative_keywords']
        self.perspective = discovery.build(
         "commentanalyzer",
         "v1alpha1",
         developerKey=self.Google_API_key,
         discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
         static_discovery=False,
        )
        self.comments_seen = 0
        self.posts_seen = 0
        self.posts_made = 0
        self.comments_made = 0

    def report_status(self):
        status = {}
        status['posts_seen'] = self.posts_seen
        status['comments_seen'] = self.comments_seen
        status['posts_made'] = self.posts_made
        status['comments_made'] = self.comments_made
        status['percent'] = round(100*(self.tally/self.config['character_budget']))
        print("READ: submissions={posts_seen}\tcomment={comments_seen}\t| WRITE: post={posts_made}\treply={comments_made}\t| SPEND={percent}%".format(**status))

    def bad_keyword(self,text):
        return [keyword for keyword in self.negative_keywords if re.search(r"\b{}\b".format(keyword), text, re.IGNORECASE)]

    def is_toxic(self,text):
        analyze_request = {
         'comment': { 'text': text },
         'requestedAttributes': {'SEVERE_TOXICITY': {}},
         'languages': 'en'
        }
        response = self.perspective.comments().analyze(body=analyze_request).execute()
        score = response['attributeScores']['SEVERE_TOXICITY']['summaryScore']['value']
        print(f"Perspective severe toxicity summary score = {score}")
        if score>self.config['toxicity_threshold']:
            return True
        else:
            return False

    def on_topic(self,text,topic_list):
        payload = {
            "inputs": text,
            "parameters": {"candidate_labels": topic_list,"multi_label": True},
            "options": {"use_cache": False, "wait_for_model": True}
        }
        if not self.check_budget(text):
            print("Not enough characters left in budget to check topic")
            return False
        self.tally += len(text)
        self.report_status()
        print(f'Checking text: {text}')
        results = query(payload, self.config['topic_classifier'], self.headers)
        if not results:
            print('Topic checking failed!')
            return False
        for k in range(len(topic_list)):
            topic = topic_list[k]
            score = results['scores'][k]
            print('{} "{}"'.format(round(score,1),topic))
            if score > self.config['topic_threshold']:
                return True
        # otherwise
        return False

    def check_budget(self,string):
        # check to see if an input string would exceed character budget
        # first, check the date; reset it and the tally if changed
        if date.today() != self.today:
            # reset the character budget and date
            self.today = date.today()
            self.tally = 0
        character_cost = len(string)
        if (self.tally + character_cost) < self.config['character_budget']:
            return True
        else:
            return False

    def describe_image(self,url):
        # Settings below for Azure vision
        headers = {
            # Request headers
            'Content-Type': 'application/json',
            'Ocp-Apim-Subscription-Key': self.Azure_token
        }

        params = urllib.parse.urlencode({
            # Request parameters
            'maxCandidates': '1',
            'language': 'en',
            'model-version': 'latest',
        })
        caption = ''
        try:
            conn = http.client.HTTPSConnection(self.config['azure_endpoint'])
            conn.request("POST", "/vision/v3.2/describe?%s" % params, '{"url":"'+url+'"}', headers)
            response = conn.getresponse()
            data = json.loads(response.read())
            #print(data)
            caption = 'A picture of ' + data['description']['captions'][0]['text']
            conn.close()
            print("Caption: "+caption)
        except Exception as e:
            print(e)
        return caption

    def generate_image(self,prompt):
        endpoint = 'https://hf.space/embed/multimodalart/latentdiffusion/+/api/predict/'
        r = requests.post(url=endpoint, json={"data": [prompt,45,'256','256',1,1]})
        r_json = r.json()
        b = base64.b64decode(r_json["data"][0].split(",")[1])
        with open("tmp.jpg", "wb") as outfile:
            outfile.write(b)
        # upscale API
        r2 = requests.post(
            "https://api.deepai.org/api/torch-srgan",
            files={
                'image': open('tmp.jpg', 'rb'),
            },
            headers={'api-key': self.DeepAI_API_key}
        )
        r2_json = r2.json()
        url = r2_json['output_url']
        return url

    def make_post(self):
        for attempt in range(self.config['post_tries']):
            # ssi-bot style GPT-2 model text post generation
            if random.random()<self.config['linkpost_share']:
                prompt = '<|sols'
            else:
                prompt = '<|soss'
            if not self.check_budget(prompt):
                print("Not enough characters left in budget to make a post!")
                return None
            self.tally += len(prompt)
            self.report_status()
            print("Generating a post on r/"+self.sub.display_name)
            post_params = self.config['post_textgen_parameters']
            stringlist = generate_text(prompt,self.config['post_textgen_model'],post_params,self.headers)
            if not stringlist:
                print("Text generation failed!")
                return None
            for generated_text in stringlist:
                print(f"GENERATED: {generated_text}")
                if self.bad_keyword(generated_text) or self.is_toxic(generated_text):
                    print("Generated text failed toxicity check, discarded.")
                    continue
                post = self.SSI.extract_submission_from_generated_text(generated_text)
                if not post:
                    print("Failed to extract post from generated text!")
                    continue
                if prompt == '<|soss':
                    if 'selftext' not in post.keys():
                        submission = self.sub.submit(title=post['title'],selftext='',flair_id=self.config['post_flair'])
                    else:
                        submission = self.sub.submit(title=post['title'],selftext=post['selftext'],flair_id=self.config['post_flair'])
                else:
                    post['url'] = self.generate_image(post['title'])
                    submission = self.sub.submit(title=post['title'],url=post['url'],flair_id=self.config['post_flair'])
                print("Post successful!")
                self.posts_made += 1
                self.report_status()
                return submission
        # if none of the posts passed the checks
        return None

    def build_post(self):
        for attempt in range(self.config['post_tries']):
            # one-shot post generation
            prompt = self.config['bot_backstory']
            prompt = '\n'.join([prompt,'Title of a Reddit post by u/{}: "'.format(self.config['bot_username'])])
            if not self.check_budget(prompt):
                print("Not enough characters left in budget to make a post!")
                return None
            self.tally += len(prompt)
            print("Generating a post on r/"+self.sub.display_name)
            # use the reply model to generate post title
            post_params = self.config['reply_textgen_parameters']
            stringlist = generate_text(prompt,self.config['reply_textgen_model'],post_params,self.headers)
            if not stringlist:
                print("Text generation failed!")
                return None
            post = {}
            for generated_text in stringlist:
                # post titles should be a single line
                truncate = generated_text.rfind('\n')
                if truncate>-1:
                    generated_text = generated_text[:truncate+1]
                cleanStr = clean_text(generated_text)
                if not cleanStr:
                    print("Invalid generation, skipping...")
                    continue
                if len(cleanStr)>300:
                    print("Generated text too long for Reddit post title, skipping")
                    continue
                print(f"GENERATED: {cleanStr}")
                if self.bad_keyword(cleanStr) or self.is_toxic(cleanStr):
                    print("Generated text failed toxicity check, discarded.")
                    continue
                post['title'] = cleanStr
            if 'title' not in post.keys():
                print("Unable to generate an acceptable post title!")
                return None
            if random.random()<self.config['linkpost_share']:
                post['url'] = self.generate_image(post['title'])
                submission = self.sub.submit(title=post['title'],url=post['url'],flair_id=self.config['post_flair'])
            else:
                prompt = prompt + post['title'] + '"'
                prompt = '\n'.join([prompt,'Post body: "'.format(self.config['bot_username'])])
                if not self.check_budget(prompt):
                    print("Not enough characters left in budget to generate post body!")
                    return None
                else:
                    self.tally += len(prompt)
                    stringlist = generate_text(prompt,self.config['reply_textgen_model'],post_params,self.headers)
                    for generated_text in stringlist:
                        cleanStr = clean_text(generated_text)
                        if not cleanStr:
                            print("Invalid generation, skipping...")
                            return None
                        print(f"GENERATED: {cleanStr}")
                        if self.bad_keyword(cleanStr) or self.is_toxic(cleanStr):
                            print("Generated text failed toxicity check, discarded.")
                            continue
                        post['selftext'] = cleanStr
                if 'selftext' not in post.keys():
                    submission = self.sub.submit(title=post['title'],selftext='',flair_id=self.config['post_flair'])
                else:
                    submission = self.sub.submit(title=post['title'],selftext=post['selftext'],flair_id=self.config['post_flair'])
                print("Post successful!")
                self.posts_made += 1
                self.report_status()
                return submission
        # if none of the posts passed the checks
        return None

    def generate_reply(self, comment):
        print("Generating a reply to comment:\n"+comment.body)
        reply = None
        # accumulate comment thread for context
        at_top = False
        prompt = 'Reply by u/{}: "'.format(self.config['bot_username'])
        thread_item = comment
        for level in range(self.config['max_levels']):
            prompt = '\n'.join(['Comment by u/{}: "{}"'.format(thread_item.author.name, thread_item.body),prompt])
            if thread_item.parent_id[:2]=='t3':
                # next thing is the post, not a comment
                # To do: image recognition/description for link posts
                at_top = True
                thread_post = comment.submission
                thread_OP = thread_post.author.name
                post_title = thread_post.title
                if thread_post.is_self:
                    post_body = thread_post.selftext
                    prompt = '\n'.join(['Post by u/{} titled "{}": "{}"'.format(thread_OP,post_title,post_body),prompt])
                else:
                    alt_text = self.describe_image(thread_post.url)
                    prompt = '\n'.join(['Image post by u/{} titled "{}": {}'.format(thread_OP,post_title,alt_text),prompt])
                break
            else:
                thread_item = thread_item.parent()
        if not at_top:
            print("Post not in prompt, discarding")
            return None
        prompt = '\n'.join([self.config['bot_backstory'],prompt])
        if not self.check_budget(prompt):
            print("Prompt is too long, skipping...")
            return None
        self.tally += len(prompt)
        self.report_status()
        print(f"PROMPT: {prompt}")
        reply_params = self.config['reply_textgen_parameters']
        stringlist = generate_text(prompt,self.config['reply_textgen_model'],reply_params,self.headers)
        if not stringlist:
            print("Generation failed, skipping...")
            return None
        for generated_text in stringlist:
            cleanStr = clean_text(generated_text)
            if not cleanStr:
                print("Invalid generation, skipping...")
                continue
            print(f"GENERATED: {cleanStr}")
            if self.is_toxic(cleanStr):
                print("Text is toxic, skipping...")
                continue
            reply = comment.reply(body=cleanStr)
            print("Reply successful!")
            self.comments_made += 1
            self.report_status()
            return reply
        return None # No valid replies

    def make_comment(self, submission):
        comment = None
        # reply to a submission
        thread_OP = submission.author.name
        post_title = submission.title
        print("Commenting on submission:\n"+post_title)
        prompt = 'Comment by u/{}: "'.format(self.config['bot_username'])
        if submission.is_self:
            post_body = submission.selftext
            prompt = '\n'.join(['Post by u/{} titled "{}": "{}"'.format(thread_OP,post_title,post_body),prompt])
        else:
            alt_text = self.describe_image(submission.url)
            prompt = '\n'.join(['Image post by u/{} titled "{}": {}'.format(thread_OP,post_title,alt_text),prompt])
        prompt = '\n'.join([self.config['bot_backstory'],prompt])
        if not self.check_budget(prompt):
            print("Prompt is too long, skipping...")
            return None
        self.tally += len(prompt)
        self.report_status()
        print(f"PROMPT: {prompt}")
        reply_params = self.config['reply_textgen_parameters']
        stringlist = generate_text(prompt,self.config['reply_textgen_model'],reply_params,self.headers)
        if not stringlist:
            print("Generation failed, skipping...")
            return None
        for generated_text in stringlist:
            cleanStr = clean_text(generated_text)
            if not cleanStr:
                print("Invalid generation, skipping...")
                return None
            print(f"GENERATED: {cleanStr}")
            if self.is_toxic(cleanStr) or self.bad_keyword(cleanStr):
                print("Text is toxic, skipping...")
            else:
                reply = submission.reply(body=cleanStr)
                print("Comment successful!")
                self.comments_made += 1
                self.report_status()
                return reply
        # no valid replies
        return None

    def watch_submissions(self):
        # watch for posts
        while True:
            for submission in self.sub.stream.submissions(pause_after=0,skip_existing=True):
                # decide whether to reply to a post
                if not submission:
                    continue
                self.posts_seen += 1
                if submission.author == self.me:
                    continue
                if self.bad_keyword(submission.title) or (submission.is_self and self.bad_keyword(submission.selftext)):
                    continue
                if self.config['linkpost_only']==2 and not submission.is_self:
                    # force reply to image posts
                    self.make_comment(submission)
                    continue
                elif self.config['linkpost_only']==1 and submission.is_self:
                    continue
                already_replied = False
                submission.comments.replace_more(limit=None)
                for comment in submission.comments:
                    if comment.author == self.reddit.user.me():
                        already_replied = True
                        break
                if already_replied:
                    continue
                if submission.is_self:
                    post_string = '\n'.join([submission.title.lower(),submission.selftext.lower()])
                else:
                    post_string = submission.title.lower()
                if self.on_topic(post_string,self.config['topic_list']):
                    print("Generating a comment on submission "+submission.id)
                    self.make_comment(submission)

    def watch_inbox(self):
        while True: # not sure if this line is necessary
            # for comment in self.reddit.inbox.unread(limit=None):
            for comment in self.reddit.inbox.stream(pause_after=0, skip_existing=True):
                if not comment:
                    continue
                if not comment.was_comment:
                    # it's actually a message - don't reply; too expensive
                    # however, shut down with a specific killphrase
                    if comment.body=="shutdown":
                        sys.exit()
                    comment.mark_read()
                    continue
                self.comments_seen += 1
                if not comment.author:
                    comment.mark_read()
                    continue
                if self.bad_keyword(comment.body):
                    print("Bad keyword found, skipping...")
                    comment.mark_read()
                    continue
                already_replied = False
                comment.replies.replace_more(limit=None)
                for reply in comment.replies:
                    if reply.author == self.me:
                        already_replied = True
                        break
                if already_replied:
                    comment.mark_read()
                    continue
                print('Checking comment "{}"'.format(comment.body))
                if comment.parent_id[:2]=='t3' and self.config['force_top_reply']:
                    self.generate_reply(comment)
                elif self.check_budget(comment.body) and words_below(comment.body, 1000):
                    if not self.config['topic_list']:
                        # get the thing to which the commenter was responding
                        comment_parent = comment.parent()
                        topic_list = get_keywords(comment_parent.body)
                        print("Parent keywords: "+", ".join(topic_list))
                    else:
                        topic_list = self.config['topic_list']
                    if self.on_topic(comment.body,topic_list):
                        self.generate_reply(comment)
                print('Comment not selected for reply, skipping...')
                comment.mark_read()

    def submission_loop(self):
        for t in self.config['post_schedule']:
            if not self.config['post_textgen_model']:
                schedule.every().day.at(t).do(self.build_post)
            else:
                schedule.every().day.at(t).do(self.make_post)
        while True:
            schedule.run_pending()
            time.sleep(1)

    def run(self):
        print("Bot named {} running on {}".format(self.config['bot_username'],self.config['bot_subreddit']))
        if not self.config['post_schedule']:
            print("No posts scheduled!")
        else:
            print("Launching submission writer")
            self.submission_writer.start()
        # don't bother running submission reader if bot has no interests
        if not self.config['topic_list']:
            print("No bot topics set, will not read submissions.")
        else:
            print("Scanning for posts on the following topics: "+", ".join(self.config['topic_list']))
            self.submission_reader.start()
        print("Launching inbox reader")
        self.inbox_reader.start()

def main():
    bot = reddit_bot(sys.argv[1]) #"bot_config.yaml"
    bot.run()

if __name__ == "__main__":
    main()
