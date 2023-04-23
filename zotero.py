import os
import requests
from pyzotero import zotero
import uuid
from supabase import create_client, Client
import json
import html
import re
import uuid
from PyPDF2 import PdfReader, PdfWriter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import openai
import tempfile
from dotenv import load_dotenv
from pathlib import Path

env_path = Path('.') / '.env.local'
load_dotenv(dotenv_path=env_path)


def validate_credentials(user_id, api_key):
    try:
        zot = zotero.Zotero(user_id, 'user', api_key)
        zot.num_items()  # This will raise an exception if the credentials are invalid
        return True
    except Exception as e:
        return False



def run_zotero(user_id, api_key):

    nltk.download('stopwords')
    nltk.download('punkt')



    Random_ID = str(uuid.uuid4())
    YOUR_USER_ID = user_id
    YOUR_API_KEY = api_key
    OUTPUT_FOLDER = os.getcwd() + "/output/"

    #if doesnt exist create folder
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)



    openai.api_key = os.getenv('OPENAI_API_KEY')
    print("hello sir")

    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_KEY')
    
    zot = zotero.Zotero(YOUR_USER_ID, 'user', YOUR_API_KEY)



    try:
        supabase = create_client(url, key)
    except Exception as e:
        print(f"Error creating Supabase client: {e}")
        exit(1)



    def strip_html_tags(text: str) -> str:
        return re.sub(r'<[^>]*>', '', text)

    def raw_text(file_path: str) -> str:
        reader = PdfReader(file_path, strict= False)
        raw_text = ''
        for page in reader.pages:
            raw_text += page.extract_text()
        return raw_text


    def split_text_into_chunks(text, max_tokens=400):
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence_tokens = word_tokenize(sentence)

            if len(current_chunk.split()) + len(sentence_tokens) <= max_tokens:
                current_chunk += sentence
            else:
                chunks.append(current_chunk)
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks


    def generate_embeddings(chunks):
        embeddings = []
        for chunk in chunks:
            response = openai.Embedding.create(
                input=chunk, model="text-embedding-ada-002"
            )
            embedding = response["data"][0]["embedding"]
            embeddings.append(embedding)
        return embeddings



    def reset_eof_of_pdf_return_stream(pdf_stream_in: list) -> list:
        for i, x in enumerate(pdf_stream_in[::-1]):
            if b'%%EOF' in x:
                actual_line = len(pdf_stream_in) - i
                break
        else:
            actual_line = len(pdf_stream_in)
            pdf_stream_in.append(b'%%EOF')
        return pdf_stream_in[:actual_line]


    def remove_js_and_save(input_path, output_path):
        with open(input_path, 'rb') as fr:
            reader = PdfReader(fr, strict=False)
            writer = PdfWriter()

            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                page._dictionary.pop("/AA", None)  # Remove JavaScript
                page._dictionary.pop("/AcroForm", None)  # Remove AcroForm data
                writer.add_page(page)

            with open(output_path, 'wb') as fw:
                writer.write(fw)
                # Ensure there is an EOF marker at the end of the file
                fw.write(b'%%EOF')


    def fix_pdf_eof(file_name):
        fixed_file_name = file_name.replace('.pdf', '') + '_fixed.pdf'
        #remove_js_and_save(file_name, fixed_file_name)
        return fixed_file_name

    def clean_text(text: list) -> str:
        #stop_words = set(stopwords.words('english'))
        cleaned_text = []
        for chunk in text:
            # Convert to lowercase
            #chunk = chunk.lower()
            
            # Remove special characters and numbers
            #chunk = re.sub(r'[^a-zA-Z]+', ' ', chunk)
            
            # Tokenize words
            word_tokens = word_tokenize(chunk)
            
            # Remove stop words and non-alphabetic words
            filtered_text = [word for word in word_tokens if word.lower()]
            
            cleaned_text.append(" ".join(filtered_text))

        return cleaned_text




    attachment_items = zot.items(itemType='attachment', format='keys', linkMode='imported_file', contentType='application/pdf').decode('utf-8')

    #print(attachment_items)

    successful_count = 0
    total_count = len(attachment_items.strip().split('\n'))



    for key in attachment_items.strip().split('\n'):
        attachment_metadata = zot.item(key, content='json')

        attachment_dict = attachment_metadata[0]

        try:
            if attachment_dict['contentType'] == 'application/pdf':
                filename = attachment_dict.get('filename', attachment_dict.get('title', f"{uuid.uuid4().hex}.pdf"))

                if attachment_dict['linkMode'] == 'imported_url':
                    pdf_url = attachment_dict['url']
                else:
                    pdf_url = zot.file(key)

                response = requests.get(pdf_url, headers={'Authorization': f'Bearer {YOUR_API_KEY}'})


                with tempfile.TemporaryDirectory() as tmp_dir:
                    output_path = os.path.join(OUTPUT_FOLDER, filename)


                    with open(output_path, 'wb') as f:
                        f.write(response.content)

                #print(f'Saved {filename} to {output_path}')

                # Fix the EOF issue
                    fix_pdf_eof(output_path)

                    raw_output = raw_text(output_path)
                    #print(raw_output)

                    chunks = split_text_into_chunks(raw_output)
                    cleaned_output = clean_text(chunks)

                    embeddings = generate_embeddings(chunks)
                    #print(embeddings)


                    successful_count += 1


                    # Get metadata
                    parent_item_key = attachment_dict['parentItem']
                    parent_item = zot.item(parent_item_key, content='json')[0]
                    #print(parent_item)

                    # Get inline citation (APA format)
                    inline_citation = zot.item(parent_item_key, content='citation', style='apa')[0]

                    # Get full citation (APA format)
                    full_citation = zot.item(parent_item_key, content='bib', style='apa')[0]

                    # Extract required fields
                    title = parent_item.get('title', '')
                    abstract = parent_item.get('abstractNote', '')
                    doi = parent_item.get('DOI', '')
                    in_text_citation = html.unescape(inline_citation)
                    full_citation_text = html.unescape(full_citation)
                    stripped_inline_citation = strip_html_tags(inline_citation)
                    stripped_full_citation = strip_html_tags(full_citation)

                    date = parent_item.get('date', '')

                    # Send data to Supabase
                    insert_data = {
                        "uuid": Random_ID,
                        'title': title,
                        'doi': doi,
                        'date': date,
                        'abstract': abstract,
                        'in_text_citation': stripped_inline_citation,
                        'full_citation': stripped_full_citation,
                    }

                    supabase_response = supabase.from_("papers").insert([insert_data]).execute()
                    #print(f"Supabase response: {json.dumps(supabase_response, indent=2)}")


                    for i, (chunk, embedding) in enumerate(zip(cleaned_output, embeddings)):
                            insert_embeddings = {
                                "doi":  doi,
                                "uuid": Random_ID,
                                "content": chunk,
                                "chunk_id": i,
                                "tokens": len(word_tokenize(chunk)),
                                "length": len(chunk),
                                "embedding": embedding
                            }
                            supabase_response = supabase.from_("embeddings").insert([insert_embeddings]).execute()

            else:
                #print(f"Skipping non-PDF attachment: {attachment_dict.get('filename', 'unknown')}")
                continue
        except Exception as e:
            print(f"Error processing attachment: {attachment_dict.get('filename', 'unknown')}, reason: {str(e)}")

    print(f"Successfully processed {successful_count} out of {total_count} papers.")

