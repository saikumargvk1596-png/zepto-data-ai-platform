import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://books.toscrape.com/"
GBP_TO_INR = 105.50

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}


def get_soup(url):
    response = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def parse_book(card):
    title_tag = card.select_one("h3 a")
    price_tag = card.select_one(".price_color")
    rating_tag = card.select_one(".star-rating")
    availability_tag = card.select_one(".availability")

    if not title_tag or not price_tag:
        return None

    title = title_tag.get("title", "").strip()

    price_text = price_tag.get_text(strip=True)
    try:
        price = float(
            price_text.replace("£", "").replace("Â", "").strip()
        )
    except (ValueError, AttributeError):
        price = None

    rating_text = None
    if rating_tag:
        classes = rating_tag.get("class", [])
        rating_text = next(
            (x for x in classes if x in RATING_MAP),
            None
        )

    rating = RATING_MAP.get(rating_text)

    availability = (
        availability_tag.get_text(" ", strip=True)
        if availability_tag
        else ""
    )

    return {
        "title": title,
        "price_gbp": price,
        "star_rating": rating_text,
        "rating": rating,
        "availability": availability,
        "in_stock": "In stock" in availability
    }


def scrape_page(url):
    soup = get_soup(url)

    rows = []

    for card in soup.select("article.product_pod"):
        book = parse_book(card)

        if book:
            rows.append(book)

    return rows


def get_category_from_book_page(book_url):
    """
    Open the individual book page and extract its category.
    """
    soup = get_soup(book_url)

    breadcrumb = soup.select("ul.breadcrumb li a")

    if len(breadcrumb) >= 3:
        return breadcrumb[-1].get_text(strip=True)

    return "Unknown"


def scrape_books(min_categories=3, minimum_books=60):

    rows = []

    # First 5 all-products pages.
    for page_number in range(1, 6):

        if page_number == 1:
            url = BASE_URL + "index.html"
        else:
            url = BASE_URL + f"catalogue/page-{page_number}.html"

        print("Scraping:", url)

        soup = get_soup(url)

        for card in soup.select("article.product_pod"):

            title_tag = card.select_one("h3 a")

            if not title_tag:
                continue

            book_url = urljoin(
                url,
                title_tag.get("href")
            )

            book = parse_book(card)

            if book:
                book["category"] = get_category_from_book_page(book_url)
                rows.append(book)

    df = pd.DataFrame(rows)

    # Remove duplicate books.
    df = df.drop_duplicates(
        subset=["title"]
    ).reset_index(drop=True)

    # Convert numeric columns.
    df["price_gbp"] = pd.to_numeric(
        df["price_gbp"],
        errors="coerce"
    )

    df["rating"] = pd.to_numeric(
        df["rating"],
        errors="coerce"
    )

    # Drop rows where price could not be parsed.
    df = df.dropna(
        subset=["price_gbp"]
    ).copy()

    # Impute missing ratings.
    if df["rating"].isna().any():
        df["rating"] = df["rating"].fillna(
            df["rating"].median()
        )

    df["rating"] = (
        df["rating"]
        .round()
        .clip(1, 5)
        .astype(int)
    )

    df["in_stock"] = (
        df["in_stock"]
        .fillna(False)
        .astype(bool)
    )

    df["price_gbp"] = df["price_gbp"].astype(float)

    # Required fixed conversion.
    df["price_inr"] = (
        df["price_gbp"] * GBP_TO_INR
    )

    # Check assignment requirements.
    if len(df) < minimum_books:
        raise RuntimeError(
            f"Only {len(df)} books were scraped. "
            f"Need at least {minimum_books}."
        )

    if df["category"].nunique() < min_categories:
        raise RuntimeError(
            f"Only {df['category'].nunique()} categories found. "
            f"Need at least {min_categories}."
        )

    print()
    print("SUCCESS")
    print("Books:", len(df))
    print("Categories:", df["category"].nunique())
    print()

    return df


if __name__ == "__main__":

    df = scrape_books()

    print(df.head())
    print()
    print(df.dtypes)
    print()
    print("Total books:", len(df))
    print("Total categories:", df["category"].nunique())