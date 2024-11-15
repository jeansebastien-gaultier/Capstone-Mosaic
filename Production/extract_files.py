import requests
from bs4 import BeautifulSoup
import sys
import pdfplumber
from io import BytesIO


def extract_text_from_pdf(pdf_url):
    pdf_file = download_pdf(pdf_url)

    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        return text


def download_pdf(url):
    response = requests.get(url)
    response.raise_for_status()  # Check if the request was successful
    return BytesIO(response.content)


def extract_pdf_url(html):
    soup = BeautifulSoup(html, "html.parser")
    links = soup.find_all("a", href=True)
    for link in links:
        url = link["href"]
        if url[-3:] == "pdf":
            return url
    return None


def google_search(query):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    }
    url = f"https://www.google.com/search?q={query}"
    response = requests.get(url, headers=headers)
    return response.text


def main(symbol):
    url = f"https://discountingcashflows.com/api/transcript/?ticker={symbol}&quarter=Q4&year=2023&key=6e9d241b-f336-4237-8935-2d70cd133969"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    }
    response = requests.get(url, headers=headers)
    with open(f"{symbol}_earning_call.txt", "w", encoding="utf-8") as file:
        file.write(response.json()[0]["content"])

    def google_search(query):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        }
        url = f"https://www.google.com/search?q={query}"
        response = requests.get(url, headers=headers)
        return response.text

    def extract_pdf_url(html):
        soup = BeautifulSoup(html, "html.parser")
        links = soup.find_all("a", href=True)
        for link in links:
            url = link["href"]
            if url[-3:] == "pdf":
                return url
        return None

    # Perform Google search and extract PDF link
    html = google_search(f"{symbol} annual report filetype:pdf")
    pdf_url = extract_pdf_url(html)

    pdf_text = extract_text_from_pdf(pdf_url)
    with open(f"{symbol}_annual report.txt", "w", encoding="utf-8") as file:
        file.write(pdf_text)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_argument = sys.argv[1]
        main(input_argument)
    else:
        print("No input argument provided")
