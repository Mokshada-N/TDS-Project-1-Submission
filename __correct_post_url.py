# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "rich",
#  "beautifulsoup4",
#   "google-genai",
# "Pillow",
# "tqdm"
# ]
# ///
import os
from glob import glob
import json
import requests
# json_file = "discourse_posts_1/topic_161071_Which_subject_to_choose_in_jan_term.json"
# BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"
# DISCOURSE_COOKIE = "5I7ZeyfSxwroQClusDlqin%2F2Qg0KxwHbcANDS05WhlQ6gAWzQHC7ePhQvj%2BNwOW%2BFGY8%2BOSfMPFT6vo3WAg5p%2Fmgt8wkMvZTo89BBdxJeSVDnTFXUML4%2F4tlq2TIYGgCQFJz4La42ZLhWnrSpYAbFSFxDzsEkoHMqbBU6ywEcgbYwa5a5SgdsUbvI%2FREBwSs20vIvU%2F5ZtVy6zE%2F4BcdFiwIW55VHNjK%2BAD8bzSVz9Qsm9A%2BO9r74VMHrAg0Bc1q%2BF2ywe98L5bxGXZOF1VDVlq%2FMbAwTspltqRh3Ct4s0dfCvQ0rrP2QPk4mguOwrW%2B--Zxq9cTiQIql0AHYj--7sHZZIy9PYnLpCoV6z%2BlQA%3D%3D"

# # Setup session
# session = requests.Session()
# session.cookies.set("_t", DISCOURSE_COOKIE, domain="discourse.onlinedegree.iitm.ac.in")
# session.headers.update({"User-Agent": "Mozilla/5.0"})
# # with open(json_file, "r", encoding="utf-8") as f:
# #     json_data = json.load(f)
    
# import requests

# BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"

# def get_topic_slug(topic_id):
#     """Fetch topic slug from Discourse API given topic_id."""
#     base_topic_url = f"{BASE_URL}/t/{topic_id}.json"
#     r = session.get(base_topic_url)
#     print(base_topic_url)
#     print(r.status_code)
#     data = r.json()
#     if r.status_code == 200:
#         data = r.json()
#         return data.get('slug')
#     else:
#         print(f"Failed to get topic slug for topic_id {topic_id}, status code: {r.status_code}")
#         return None

# def update_post_urls(json_data):
#     """Replace the post_url field in each post with the correct format."""
#     topic_id = json_data["topic_id"]
#     posts = json_data["posts"]
#     topic_slug = get_topic_slug(topic_id)
#     if not topic_slug:
#         print("Cannot update URLs without topic_slug")
#         return json_data
#     for post in posts:
#         post_number = post.get('post_number')
#         # Overwrite the post_url field
#         post['post_url'] = f"{BASE_URL}/t/{topic_slug}/{topic_id}/{post_number}"
#         post['topic_url'] = f"{BASE_URL}/t/{topic_slug}/{topic_id}"
        
#     return json_data

# # Directory containing your JSON files
# json_dir = "discourse_posts_1"

# # Use glob to find all .json files in the directory (non-recursive)
# json_files = glob(os.path.join(json_dir, "*.json"))

# # If you want to search recursively in subfolders, use:
# # from glob import glob
# # json_files = glob(os.path.join(json_dir, "**", "*.json"), recursive=True)

# for json_file in json_files:
#     print(f"Processing {json_file}...")
#     try:
#         with open(json_file, "r", encoding="utf-8") as f:
#             json_data = json.load(f)
#         updated_json = update_post_urls(json_data)
#         with open(json_file, "w", encoding="utf-8") as f:
#             json.dump(updated_json, f, ensure_ascii=False, indent=2)
#         print(f"Updated URLs in {json_file}")
#     except Exception as e:
#         print(f"Error processing {json_file}: {e}")

# # updated_json = update_post_urls(json_data)

# # for post in updated_json["posts"]:
# #     print(post["post_url"])

# # with open(json_file, "w", encoding="utf-8") as f:
# #     json.dump(updated_json, f, ensure_ascii=False, indent=2)

# # topic_id = topic_data["topic_id"]
# #     base_topic_url = f"{BASE_URL}/t/{topic_id}.json"
# #     r = session.get(base_topic_url)
# #     print(base_topic_url)
# #     print(r.status_code)
# #     data = r.json()
# #     topic_slug = data.get("slug")
# #     print(topic_slug)
# #     posts = topic_data["posts"]
# #     for post in posts:
# #         post_url = f"{BASE_URL}/t/{topic_slug}/{topic_id}/{post['post_number']}"
# #         print(post_url)
# #         post["post_url"] = post_url



# NEW
import os
import json
import requests
from glob import glob

BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"
DISCOURSE_COOKIE = "%2F4lTmJrfcWQjmxOtYu8UcNmEzR5w5LdVclkSiyeS7pAIxaNrt3y67nt%2BZmUdCn2dtwSPDwlEEFM4Mn6Dxp1jX8yWGlC9rnDHir6y7N6rxRCAQbBid8DXTpkDwatyx3bqCJMsQvDLFRd7TvCY6XUN%2B%2BBSW5i1Ku0Jnk7yq3kHlYlfpVFcF2QpDkUn%2FUaiG%2FMd9VNGLJGx8PKzYbeVFx62QAuBFAnGwQNPHBFlcXygIxk4KyXulPcMHKB4xyUqX4KEOncHkSwzjv4r%2BVEvQkllX40PPPJYPWKLF59XK%2BE6n%2B3Slr3ManqTFH%2BB3Wk0i32F--PyZMCXqlbCopfC0p--kOXs%2FhDmeqIqY0QFFfVimA%3D%3D"  # Replace with your actual cookie

session = requests.Session()
session.cookies.set("_t", DISCOURSE_COOKIE, domain="discourse.onlinedegree.iitm.ac.in")
session.headers.update({"User-Agent": "Mozilla/5.0"})

def fetch_posts_for_topic(topic_id):
    url = f"{BASE_URL}/t/{topic_id}.json"
    r = session.get(url)
    if r.status_code == 200:
        data = r.json()
        # Return a list of post dicts
        return data.get('post_stream', {}).get('posts', []), data.get('slug')
    else:
        print(f"Failed to get posts for topic_id {topic_id}, status code: {r.status_code}")
        return [], None

json_dir = "discourse_posts_updated"
json_files = glob(os.path.join(json_dir, "*.json"))
# json_files = ["discourse_posts_updated/topic_171672_Why_Failed_.json"]

for json_file in json_files:
    print(f"Processing {json_file}...")
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        topic_id = json_data["topic_id"]
        api_posts, topic_slug = fetch_posts_for_topic(topic_id)
        if not api_posts or not topic_slug:
            print(f"Skipping {json_file} (could not fetch posts/slug)")
            continue
        # Add/update topic_url at the root
        json_data["topic_url"] = f"{BASE_URL}/t/{topic_slug}/{topic_id}"
        # For each post in your local JSON, find the matching post in the API response by post_number
        for post in json_data.get("posts", []):
            post_number = post.get("post_number")
            # Find the API post with the same post_number
            api_post = next((p for p in api_posts if p["post_number"] == post_number), None)
            if api_post:
                post["reply_to_post_number"] = api_post.get("reply_to_post_number")
            else:
                post["reply_to_post_number"] = None  # Could not find, set as None
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"Added topic_url and reply_to_post_number to {json_file}")
    except Exception as e:
        print(f"Error processing {json_file}: {e}")