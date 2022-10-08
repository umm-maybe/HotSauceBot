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
from praw.models import Message as praw_Message
from transformers import pipeline
# from detoxify import Detoxify

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
    # look for double-quotes
    truncate = generated_text.find('"')
    if truncate>-1:
        cleanStr = generated_text[:truncate]
        return cleanStr
    # if we can't find double-quotes, look for the last newline
    truncate = generated_text.rfind('\n')
    if truncate>-1:
        cleanStr = generated_text[:truncate]
        return cleanStr
    # if we can't find a newline, look for the last terminal punctuation or start of new post
    if re.search(r'[?.!(Reply|Post)]', generated_text):
        trimPart = re.split(r'[?.!(Reply|Post)]', generated_text)[-1]
        cleanStr = generated_text.replace(trimPart,'')
        return cleanStr
    # if we can't find a newline, use the last space
    truncate = generated_text.rfind(' ')
    if truncate>-1:
        cleanStr = generated_text[:truncate]
        # using the last space may result in a trailing comma or colon; remove it
        if cleanStr[-1] in ',;:':
            cleanStr = cleanStr[:-1]
        return cleanStr
    # if we can't even find any spaces, give up
    return None

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
        self.bot_backstory = self.config['bot_backstory']
        if self.config['topic_list']:
            self.topic_list = self.config['topic_list']
        else:
            self.topic_list = get_keywords(self.bot_backstory)
        self.HF_key = os.environ[self.config['HF_key_var']]
        self.headers = {"Authorization": "Bearer "+self.HF_key}
        self.DeepAI_API_key = os.environ[self.config['deepai_api_key_var']]
        self.Google_API_key = os.environ[self.config['Google_API_key_var']]
        self.Azure_token = os.environ[self.config['azure_token_var']]
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
         'requestedAttributes': {'TOXICITY': {}},
         'languages': 'en'
        }
        try:
            response = self.perspective.comments().analyze(body=analyze_request).execute()
        except:
            print("Toxicity checking failed!")
            return True
        score = response['attributeScores']['TOXICITY']['summaryScore']['value']
        print(f"Perspective toxicity summary score = {score}")
        if score>self.config['toxicity_threshold']:
            return True
        else:
            return False

    def on_topic(self,text,topic_list):
        classifier = pipeline("zero-shot-classification", self.config['topic_classifier'])
        sequence = text
        interest_prob = classifier(sequence, topic_list, multi_label=True)
        score = sum(interest_prob['scores'])/len(topic_list)
        rdraw = random.random()
        if rdraw < score:
            print('Random draw {} < average score {}'.format(round(rdraw,2),round(score,2)))
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
        r = requests.post(url=endpoint, json={"data": [prompt,50,'256','256',1,1]})
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
        if not self.config['post_textgen_model']:
            # if no fine-tuned model is given for posts, use the one-shot reply model
            submission = self.build_post()
            return submission
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
            prompt = self.bot_backstory
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
                try:
                    submission = self.sub.submit(title=post['title'],url=post['url'],flair_id=self.config['post_flair'])
                    return submission
                except:
                    print("Post unsuccessful...")
                    continue
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
                            continue
                        print(f"GENERATED: {cleanStr}")
                        if self.bad_keyword(cleanStr) or self.is_toxic(cleanStr):
                            print("Generated text failed toxicity check, discarded.")
                            continue
                        post['selftext'] = cleanStr
                if 'selftext' not in post.keys():
                    try:
                        submission = self.sub.submit(title=post['title'],selftext='',flair_id=self.config['post_flair'])
                    except:
                        print("Post unsuccessful...")
                        continue
                else:
                    try:
                        submission = self.sub.submit(title=post['title'],selftext=post['selftext'],flair_id=self.config['post_flair'])
                    except:
                        print("Post unsuccessful...")
                        continue
                print("Post successful!")
                self.posts_made += 1
                self.report_status()
                return submission
            time.sleep(900) # wait fifteen minutes
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
        # if not at_top:
        #     print("Post not in prompt, discarding")
        #     return None
        prompt = '\n'.join([self.bot_backstory,prompt])
        if not self.check_budget(prompt):
            print("Prompt is too long, skipping...")
            return None
        self.tally += len(prompt)
        self.report_status()
        print(f"PROMPT: {prompt}")
        reply_params = self.config['reply_textgen_parameters']
        try:
            stringlist = generate_text(prompt,self.config['reply_textgen_model'],reply_params,self.headers)
        except:
            print("Generation failed, skipping...")
            return None
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
            reply = comment.reply(body=clean_text(cleanStr)) # sometimes need a 2nd wash
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
        prompt = '\n'.join([self.bot_backstory,prompt])
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
                try:
                    reply = submission.reply(body=cleanStr)
                    print("Comment successful!")
                    self.comments_made += 1
                    self.report_status()
                    return reply
                except:
                    print("Comment failed, sorry...")
                    return None
        # no valid replies
        return None

    def watch_submissions(self):
        # watch for posts
        while True:
            try:
                for submission in self.sub.stream.submissions(pause_after=0,skip_existing=True):
                    # decide whether to reply to a post
                    if not submission:
                        continue
                    self.posts_seen += 1
                    if submission.author == self.me:
                        continue
                    if self.bad_keyword(submission.title) or (submission.is_self and self.bad_keyword(submission.selftext)):
                        continue
                    if self.is_toxic(submission.title) or (submission.is_self and self.is_toxic(submission.selftext)):
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
                    if self.on_topic(submission.title,self.topic_list):
                        print("Generating a comment on submission "+submission.id)
                        self.make_comment(submission)
            except:
                print("PRAW error, restarting")

    def watch_inbox(self):
        while True: # not sure if this line is necessary
            try:
                for item in self.reddit.inbox.stream(pause_after=0, skip_existing=True):
                    if not item:
                        continue
                    if isinstance(item, praw_Message):
                        # it's actually a message
                        # if item.author.name==self.config['bot_operator'] and (self.config['kill_phrase'] in item.body):
                        #     item.mark_read()
                        #     self.shutdown()
                        if self.config['dynamic_prompt']:
                            if item.subject and item.body:
                                if self.is_toxic(item.subject):
                                    item.reply(body="Backstory is toxic, rejected...")
                                    continue
                                self.bot_backstory = 'u/{} is {}'.format(self.config['bot_username'], item.subject)
                                user_topic_list = item.body.split(',')[:10]
                                if user_topic_list:
                                    self.topic_list = user_topic_list
                                else:
                                    self.topic_list = get_keywords(self.bot_backstory)
                                status = 'Backstory changed to: {} with interests {}'.format(self.bot_backstory,self.topic_list)
                                print(status)
                                item.reply(body=status)
                                self.me.subreddit.submit(title='Bot updated by {}'.format(item.author.name),selftext=status)
                                self.make_post()
                        item.mark_read()
                        continue
                    self.comments_seen += 1
                    if not item.author:
                        item.mark_read()
                        continue
                    if self.bad_keyword(item.body):
                        print("Bad keyword found, skipping...")
                        item.mark_read()
                        continue
                    if self.is_toxic(item.body):
                        print("Comment is toxic, skipping...")
                        item.mark_read()
                        continue
                    already_replied = False
                    item.replies.replace_more(limit=None)
                    for reply in item.replies:
                        if reply.author == self.me:
                            already_replied = True
                            break
                    if already_replied:
                        item.mark_read()
                        continue
                    print('Checking comment "{}"'.format(item.body))
                    if item.parent_id[:2]=='t3' and self.config['force_top_reply']:
                        self.generate_reply(item)
                    elif self.check_budget(item.body) and words_below(item.body, 1000):
                        if item.was_comment:
                            # get the keywords of the thing to which the commenter was responding
                            item_parent = item.parent()
                            if item.parent_id[:2]=='t3':
                                topic_list = get_keywords(item_parent.title)
                            else:
                                topic_list = get_keywords(item_parent.body)
                            print("Parent keywords: "+", ".join(topic_list))
                        else:
                            # only possible option here is a mention in a submission
                            if not self.topic_list:
                                topic_list = get_keywords(self.bot_backstory)
                                print("Backstory keywords: "+", ".join(topic_list))
                            else:
                                topic_list = self.topic_list
                        if self.on_topic(item.body,topic_list):
                            self.generate_reply(item)
                    print('Comment not selected for reply, skipping...')
                    item.mark_read()
            except:
                print("PRAW error, restarting")

    def submission_loop(self):
        for t in self.config['post_schedule']['mon']:
            schedule.every().monday.at(t).do(self.make_post)
        for t in self.config['post_schedule']['tue']:
            schedule.every().tuesday.at(t).do(self.make_post)
        for t in self.config['post_schedule']['wed']:
            schedule.every().wednesday.at(t).do(self.make_post)
        for t in self.config['post_schedule']['thu']:
            schedule.every().thursday.at(t).do(self.make_post)
        for t in self.config['post_schedule']['fri']:
            schedule.every().friday.at(t).do(self.make_post)
        for t in self.config['post_schedule']['sat']:
            schedule.every().saturday.at(t).do(self.make_post)
        for t in self.config['post_schedule']['sun']:
            schedule.every().sunday.at(t).do(self.make_post)
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
        if self.config['read_posts']:
            print("Scanning for posts on the following topics: "+", ".join(self.topic_list))
            self.submission_reader.start()
        else:
            print("Bot will not read submissions.")
        print("Launching inbox reader")
        self.inbox_reader.start()

    def shutdown(self):
        sys.exit()

def main():
    bot = reddit_bot(sys.argv[1]) #"bot_config.yaml"
    bot.run()

if __name__ == "__main__":
    main()
