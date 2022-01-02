#Import libraries
import pytesseract
from PIL import Image, ImageFont, ImageDraw 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import matplotlib.pyplot as plt
import keras_ocr
import cv2 
import math
import numpy as np
import gradio as gr
import numpy as np
#Support for Hindi, Spanish, French, Arabic, Turish, Gailec/Irish, and German
#'hindi':
tokenizerhi = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
modelhi = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
#'spanish':
tokenizeres = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
modeles = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-es")
#'german':
tokenizerde = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
modelde = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")
#'french':
tokenizerfr = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
modelfr = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
#'turkish':
tokenizertrk = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-trk")
modeltrk = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-trk")
#'arabic':
tokenizerar = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
modelar = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
#Irish /Gaelish
tokenizerga = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ga")
modelga = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ga")
#Translate in your desired language
def choose_language(language):
  #Loading the tokenizers and trained models
  if language == 'hindi':
    tokenizer, model = tokenizerhi, modelhi
  elif language == 'spanish':
    tokenizer, model = tokenizeres, modeles
  elif language == 'german':
    tokenizer, model = tokenizerde, modelde
  elif language == 'french':
    tokenizer, model = tokenizerfr, modelfr
  elif language == 'turkish':
    tokenizer, model = tokenizertrk, modeltrk
  elif language == 'arabic':
    tokenizer, model = tokenizerar, modelar
  else:
    tokenizer, model = tokenizerga, modelga
  return tokenizer, model
#Function to translate english text to desired language
def translator(text, lang):
  if '\n' in text:
    text_list = text.splitlines()
    text = ' '.join(text_list)
  #Huggingface transformers Magic 
  tokenizer, model = choose_language(lang)
  input_ids = tokenizer.encode(text, return_tensors="pt", padding=True)  #Tokenizer
  outputs = model.generate(input_ids)  #Model
  #Translated Text
  decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)  #Tokenizer
  return decoded_text
#Getting cordinates
def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)
pipeline = keras_ocr.pipeline.Pipeline()
#Getting cordinates for text insie image
#This will help in filling up the space with colors
def img_text_cords(im): #, pipeline):
    #read image
    img = keras_ocr.tools.read(im)
    #generate (word, box) tuples 
    prediction_groups = pipeline.recognize([img])  
    mask = np.zeros(img.shape[:2], dtype="uint8")
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1] 
        x2, y2 = box[1][2]
        x3, y3 = box[1][3] 
                
        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)
        
        thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))
        
        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255,    
        thickness)
        img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)
                 
    return img 
#Extracting text from image
def text_extract(im):
  #Using pytesseract to read text
  ocr_text = pytesseract.image_to_string(im)
  return ocr_text
#Formatting the text to multi lines structure
#This is mainly for translated text to look and fit better on an image
def format_text(language,extracted_text):
  
  translated_text = translator(extracted_text, language)
  
  word_list,i = [],0
  for word in translated_text.split():
    if i%5 != 0:
      word_list.append(' '+word)
    else:
      word_list.append('\n'+word)
    i+=1 
  new_title_text = ''.join(word_list)
  return new_title_text
def translate_image(im, language):
  #Extract text, translate in your language and format it 
  extracted_text = text_extract(im)
  #font select -- Getting Unicode Text
  title_font = ImageFont.truetype('./arial-unicode-ms.ttf',30) 
  #text to write on image #Example in hindi - Unicode text u"आप जीवन में मिलता हर मौका ले लो, क्योंकि कुछ चीजें केवल एक बार होती हैं. शुभ सुबह"  
  txt = format_text(language,extracted_text)
  #Editing image
  img_returned = img_text_cords(im) 
  img_rgb = cv2.cvtColor(img_returned, cv2.COLOR_BGR2RGB)
  cv2.imwrite("text_free_image.jpg",img_rgb)
  new_image = Image.open("text_free_image.jpg")
  #Enable writing on image
  image_editable = ImageDraw.Draw(new_image)
  image_editable.multiline_text((10,10), txt,spacing=2, font=title_font, fill= (237, 230, 211))  # Text color e.g. (0, 0, 0)) blacks
  return new_image
title = "Translate English Text to Your Regional Language In Your Forwarded Images"
description = "This fun Gradio demo is for translating English quote in an image (usually whatsapp forwards :) ) to your local or preferred language. To use it, simply upload your image, select one of the language choices given (hindi, spanish, german, french, arabic, irish, and turkish) from radio buttons provided. You can alternately click one of the examples to load them and select the language choice along with it."
article = "<div style='text-align: center;'>Image Text Translate by <a href='https://twitter.com/yvrjsharma' target='_blank'>Yuvraj S</a> | <a href='https://github.com/yvrjsharma/HugginFace_Gradio' target='_blank'>Github Repo</a> | <center><img src='https://visitor-badge.glitch.me/badge?page_id=ysharma/TranslateQuotesInImageForwards' alt='visitor badge'></center></div>"
pipeline = keras_ocr.pipeline.Pipeline()
gr.Interface(
    translate_image, 
    [gr.inputs.Image(type="filepath", label="Input"), gr.inputs.Radio(choices=['hindi','spanish','french','turkish','german','irish', 'arabic'], type="value", default='hindi', label='Choose A Language')], 
    gr.outputs.Image(type="pil", label="Output"),
    title=title,
    description=description,
    article=article,
    examples=[['quote1.jpg','german'], ['en2.jpg','hindi'],['gm1.jpg','french'],['quotes6.jpg','spanish']],
    enable_queue=True
   ).launch(debug=True)
