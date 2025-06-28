import requests
import json
from bs4 import BeautifulSoup


def fetch_objects(path="people.html"):
    response = requests.get("https://cse.ucdenver.edu/~bdlab/" + path, headers={"User-Agent": "BDLab-Agent-Scraper/1.0"}, timeout=10)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")

def scrape_members():
    """
    Scrape lab members from the people page, members have the following structure: 
    <div id="team-member-cards" class="row mb-4">
        <div class="col-lg-3 col-md-6 mb-4 card-small">
            <div class="card shadow-sm h-100 text-center">
            <img … />
            <div class="card-body">
                <h5 class="card-title">Name</h5>
                <p class="card-text">Role</p>
                <p class="card-text"><b>Research Interests:</b> …</p>
                <p class="card-text"><b>Projects:</b> …</p>
                <div class="social-icons mt-3">…</div>
            </div>
            </div>
        </div>
        </div>
    """
    soup = fetch_objects()
    container = soup.select_one("#team-member-cards")
    cards = container.select(".card-small")

    all_members = []
    
    #based on the structure of the page, we can start scraping all that data from the cards
    for col in cards:

        body = col.select_one(".card-body")
        if not body:
            continue

        member = {}
        name_tag = body.select_one(".card-title")

        if name_tag:
            member["name"] = name_tag.get_text(strip=True)
        else: 
            member["name"] = "Unknown"

        # look for the paragraphs in the card body
        paragraph_tags = body.find_all("p", recursive=False)

        if paragraph_tags:
            member["role"] = paragraph_tags[0].get_text(strip=True)
        else:
            member["role"] = "Unknown"

        # Body contains labeled fields like Research Interests and Projects. We extract them here
        for p in body.find_all("p"):
            b = p.find("b")

            if b and ":" in b.text:
                label = b.text.rstrip(":").lower().replace(" ", "_")
                value = p.get_text(strip=True).replace(b.text, "").strip()
                member[label] = value

        # social links
        media_links = []
        for a in body.select(".social-icons a"):
            href = a.get("href")

            if href:
                media_links.append(href)

        member["links"] = media_links
    
        all_members.append(member)

    return all_members


def scrape_research():
    """
    Scrape research topics from the research page, topics have the following structure...
    """
    topics = []
    # to scrape reserach


if __name__ == "__main__":
    try:
        data = {"members": scrape_members()}
    except Exception as e:
        print("Error while scraping:", e)

    with open("memory.json", "w") as f:
        json.dump(data, f, indent=2)

    print("Scraped and saved to memory.json")

