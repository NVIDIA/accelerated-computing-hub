import re
import requests
import os
from urllib.parse import urlparse

def find_md_links(md_content):
    """
    Extracts all Markdown links (inline and reference style) from the content.
    Returns a list of URLs.
    """
    # Regex for inline links: [text](url)
    INLINE_LINK_RE = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
    inline_links = [url for text, url in INLINE_LINK_RE.findall(md_content)]

    # Regex for reference links: [text][id] and [id]: url
    FOOTNOTE_LINK_TEXT_RE = re.compile(r'\[([^\\]]+)\]\[(\d+)\]')
    FOOTNOTE_LINK_URL_RE = re.compile(r'\[(\d+)\]:\s+(\S+)')
    footnote_links_text = dict(FOOTNOTE_LINK_TEXT_RE.findall(md_content))
    footnote_urls_map = dict(FOOTNOTE_LINK_URL_RE.findall(md_content))
    
    reference_links = []
    for key in footnote_links_text.keys():
        if footnote_links_text[key] in footnote_urls_map:
            reference_links.append(footnote_urls_map[footnote_links_text[key]])

    return inline_links + reference_links

def check_link(url, base_path="."):
    """
    Checks if a given URL is valid.
    Handles both external HTTP(S) links and local file paths.
    """
    parsed_url = urlparse(url)

    if parsed_url.scheme in ('http', 'https'):
        # Check external link
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return True, "OK"
            else:
                return False, f"Broken link (Status Code: {response.status_code})"
        except requests.exceptions.RequestException as e:
            return False, f"Error reaching URL: {e}"
    elif parsed_url.scheme in ('', 'file'):
        # Check local link
        local_path = os.path.normpath(os.path.join(base_path, parsed_url.path.strip('/')))
        if os.path.exists(local_path):
            return True, "OK"
        else:
            return False, f"Missing local file: {local_path}"
    else:
        # Other schemes (e.g., mailto, ftp) are considered valid for this test
        return True, f"Scheme {parsed_url.scheme} not checked"

def validate_links(links, base_dir, file_path):
    """
    Validates a list of links and prints the results.
    Returns True if all links are valid, False otherwise.
    """
    broken_links_found = False

    if not links:
        print(f"No links found in {file_path}.")
        return True

    print(f"Found {len(links)} links in {file_path}. Checking them...")

    for url in links:
        is_valid, message = check_link(url, base_dir)
        if not is_valid:
            print(f"  [❌ BROKEN] Link: '{url}' -> {message}")
            broken_links_found = True
        else:
            print(f"  [✅ VALID] Link: '{url}'")

    if broken_links_found:
        print(f"\nLink validation failed for {file_path}.")
        return False
    else:
        print(f"\nAll links in {file_path} are valid.")
        return True

