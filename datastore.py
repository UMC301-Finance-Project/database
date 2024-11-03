from io import BytesIO
import requests
import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModel
import torch
import firebase_admin
from firebase_admin import credentials, firestore
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging
from tqdm import tqdm
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.firefox.service import Service

def extract_tables(soup):
    tables = []
    
    # Find all table elements
    for table in soup.find_all('table'):
        table_data = []
        # Iterate through each row in the table
        for row in table.find_all('tr'):
            row_data = []
            # Get all cells in the row
            for cell in row.find_all(['td', 'th']):  # Include both data and header cells
                cell_data = cell.get_text(strip=True)  # Strip whitespace
                
                # Check if there are any links within the cell
                links = cell.find_all('a', href=True)
                if links:
                    # Append the href attribute of the first link
                    link_data = ', '.join(link['href'] for link in links)
                    cell_data += f"{link_data}"
                
                row_data.append(cell_data)
            
            if row_data:  # Only add non-empty rows
                table_data.append(row_data)
        tables.append(table_data)
    
    return tables
URLs = {"TATAMOTORS": "https://www.bseindia.com/stock-share-price/tata-motors-ltd/tatamotors/500570/",
        "TCS":"https://www.bseindia.com/stock-share-price/TCS/TCS/532540/"}
def extract_soup(url):
    service = Service('./geckodriver.exe')
    options = webdriver.FirefoxOptions()
    options.headless = True
    driver = webdriver.Firefox(service=service, options=options)
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    driver.quit()
    return soup
def extract_data(stock):
    url = URLs[stock]
    soup = extract_soup(url)
    tables = extract_tables(soup)
    data_stock = {}
    for i in range(len(tables)):
        try:
            table = tables[i]
            check = table[0][0]
            if check.startswith("Industry Classification"):
                temp = dict()
                for j in range(2, len(table),2):
                    temp[table[j][0]] = table[j][-1]
                data_stock["Detailed Industry Classification"] = temp
            elif check.startswith("High Lows"):
                temp = dict()
                for j in range(1, len(table),2):
                    temp[table[j][0]] = table[j][-1]
                data_stock["Detailed High Lows"] = temp
            elif check==("Previous Close"):
                temp = dict()
                for j in range(0, len(table),2):
                    temp[table[j][0]] = table[j][-1]
                data_stock["Previous Day"] = temp
            elif check==("52 Wk High"):
                temp = dict()
                for j in range(0, len(table),2):
                    temp[table[j][0]] = table[j][-1]
                data_stock["Price Bands"] = temp
            elif check == 'TTQ (Lakh)':
                temp = dict()
                for j in range(0, len(table),2):
                    temp[table[j][0]] = table[j][-1]
                data_stock["Market Capitalization"] = temp
            elif check == 'EPS (TTM)':
                temp = dict()
                for j in range(0, len(table),2):
                    temp[table[j][0]] = table[j][-1]
                data_stock["Financials"] = temp
            elif check == 'Category':
                temp = dict()
                for j in range(0, len(table)-1,2):
                    temp[table[j][0]] = table[j][-1]
                data_stock["Classification"] = temp
            elif check == 'Peer Group':
                try:
                    peers = table[1][1:]
                    temp = dict.fromkeys(peers)
                    for peer in temp:
                        temp[peer] = dict()
                    for j in range(3, len(table)-1, 2):
                        row = table[j]
                        heading = row[0]
                        if heading.startswith("Result"):
                            heading = "Result Date"
                        for k in range(1, len(row)):
                            temp[peers[k-1]][heading] = row[k]
                        
                    data_stock["Peer Comparison"] = temp
                except Exception as e:
                    print(e)      
        except:
            continue
    report_bse_url = url + "financials-annual-reports/"
    soup = extract_soup(report_bse_url)
    tables = extract_tables(soup)
    report_url = tables[-1][1][-1].strip()
    data_stock["report_url"] = report_url
    return data_stock

# Initialize Firestore DB
cred = credentials.Certificate("firebase_credentials.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Step 1: Download PDFs


def setup_logger():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def get_firefox_headers():
    """Return headers that mimic Firefox browser"""
    return {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0'
    }

def create_session_with_retries():
    """Create a requests session with retry strategy"""
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504]
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))
    session.headers.update(get_firefox_headers())
    return session
def extract_text_from_pdf_url(url):
    """
    Extract text from a PDF URL without downloading to disk
    
    Args:
        url (str): URL of the PDF file
        
    Returns:
        str: Extracted text from the PDF
        
    Raises:
        requests.exceptions.RequestException: If download fails
        fitz.FileDataError: If PDF processing fails
    """
    logger = setup_logger()
    session = create_session_with_retries()
    
    try:
        # Get PDF content with progress bar
        response = session.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        # Download PDF into memory
        pdf_data = BytesIO()
        with tqdm(total=total_size, unit='iB', unit_scale=True,
                 desc=f"Processing PDF from {url.split('/')[-1]}") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    pdf_data.write(chunk)
                    pbar.update(len(chunk))
        
        # Reset BytesIO position
        pdf_data.seek(0)
        
        # Extract text from PDF in memory
        pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
        text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        
        logger.info(f"Successfully extracted text from PDF at {url}")
        return text
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading PDF: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise
    finally:
        if 'pdf_document' in locals():
            pdf_document.close()

def extract_texts_from_urls(urls):
    """
    Extract text from multiple PDF URLs
    
    Args:
        urls (list): List of PDF URLs
        
    Returns:
        list: List of extracted texts
    """
    texts = []
    for url in urls:
        try:
            text = extract_text_from_pdf_url(url)
            print(len(text))
            texts.append(text)
        except Exception as e:
            print(e)
    return texts
# Step 3: Vectorize Text
def vectorize_texts(texts, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy().tolist())
    return embeddings

# Step 4: Store Data in Firestore
def store_data_in_firestore(stocks,urls, texts, embeddings, chunk_size=1000000):
    for stock,url, text, embedding in zip(stocks,urls, texts, embeddings):
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        chunk_refs = []
        for chunk in chunks:
            chunk_ref = db.collection('pdf_text_chunks').add({'chunk': chunk})
            chunk_refs.append(chunk_ref[1].id)  # Store the document ID of each chunk
        data = {
            'stock': stock,
            'url': url,
            'text_chunks': chunk_refs,
            'embedding': embedding
        }
        db.collection('pdf_documents').add(data)

# Main execution

def pipeline(data):
    stocks = list(data.keys())
    urls = [data[stock]["report_url"] for stock in stocks]
    texts = extract_texts_from_urls(urls)
    embeddings = vectorize_texts(texts)
    store_data_in_firestore(stocks,urls, texts, embeddings)