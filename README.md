# How To Deploy Your Machine Learning Models Online Using Huggingface & Gradio Powered Spaces
![](/images/gradioapp3.jpg)

## Inspiration 
I built a fun little ML powered app over the Christmas holidays, hope you have fun using it as much as I did while building it. User Research (*LOL*): The inspiration of building a tool like this comes from my mother's need of understanding *Whatsapp image forwards* which have English text written over them. I believe there are others as well who face the same struggle while trying to understand the daily fowards which are not in their regional language. Egro, added the support for 6 international languages. 

Spaces is a new and extremly useful tool to deploy or showcase your ML apps to the world. You can refer these videos - [Build and Deploy a Machine Learning App in 2 Minutes](https://www.youtube.com/watch?v=3bSVKNKb_PY) or [Building Machine Learning Applications Fast](https://www.youtube.com/watch?v=c7mle2yYpwQ&t=738s) released by Huggingface, to get more idea about it. Also, please refer this wonderful [blogpost](https://huggingface.co/blog/gradio-spaces) on how you can use HuggingFace Spaces and Gradio in matter of few lines of code. 

In this article I'll try and explain how I build this fun app and how you can build one too. Let's Go!


## Table of Content
1. HuggingFace introduction
2. Gradio introducttion
3. What I built
4. How I built it
5. How you can access the app
6. Conclusion



## Huggingface
HuggingFace is a startup in the AI field, and there mission is to democratize good machine learning. Its an AI community trying to build the future in which everyone has equal opportunity and access to benfits of latest advances in AI. You can either browse their [model hub](https://huggingface.co/models) to discover, experiment and contribute to new sota models, for example, [gooogle-tapas](https://huggingface.co/google/tapas-base), [distilbert](https://huggingface.co/distilbert-base-uncased), [facebook-wav2vec2](https://huggingface.co/facebook/wav2vec2-base-960h), and so on. Or you can directly use their inference API to serve your moddels directly from HF infrastructure. The most famous artifact that HF has created so far is their Transformer library, which started as an nlp library but now has support for other modalities as well. Now, Transformer provides thousands of pretrained models to perform tasks on different modalities such as text, vision, and audio.


## Gradio 
[Gradio](https://gradio.app/) is the fastest and easiest way to demo your ML models with a very friendly and feature-rich UI. Almost anyone can us it without a manual and with just a little intuition. You can install Gradio library easily using pip install. I used both Hugging Face and Gradio on Colab so installations were allthemore starightforward and easier. You can deploy your ML model online using just 5-10 code lines depending on the complexity of your implementation. Recently, Gradio has made it possible to embed your deployed ML model to any webpage of your choice. I have done the same at the end of this article, check it out. Gadio code helps you generate a public link for your deployed ML model/app which you can then share with your friends, colleagues at work or a potential employer or collaborator.  


## What I Built 
I built a fun project in last couple days using HuggingFace and Gradio functionalities. This project employs mage analysis, language translation and OCR techniques. A user can select an image of his choice with some english text over it as an input. For example, an image with some motivational text written over it like the ones we all receive in our family whatsapp groups all the time. He then gets to make a selection from the given 7 languages as the output language - German, Spanish, French, Turkish, Hindi, Arabic, and Irish. The app then outputs the same image as input but with text now translated in the language selected by the user.


## How I Built It
I am using pytesseract to perform the OCR on input image. Once I have the text 'extracted' from the input image, I employ HuggingFace transformers library to get the desired translation model and tokenizer loaded for an inference. These translation models are open sourced by the [Language Technology Research Group at the University of Helsinki](https://blogs.helsinki.fi/language-technology/), and you can access their account page and pre-trained  odels on HuggingFace'e [website](https://huggingface.co/Helsinki-NLP). The extracted text is then translated into the selected language. For example, if you have selected the language as German, the app will load the "Helsinki-NLP/opus-mt-en-de" translation model from transformers hub and would tranlate the OCR extracted English text to German.

Next, I am using [Kers-OCR](https://github.com/faustomorales/keras-ocr) library to extract the cordinates of English text from the original input image. This library is based on Keras CRNN or [Keras implementation of Convolutional Recurrent Neural Network for Text Recognition](https://github.com/janzd/CRNN). Once I have these cordinates, I perform a cleaning of text using OpenCV Pillow library with just couple lines of code. This cleaning is inspired from [Carlo Borella's incredible post](https://towardsdatascience.com/remove-text-from-images-using-cv2-and-keras-ocr-24e7612ae4f4).

After this, next step is to copy the translated text onto the 'cleansed' image. Current implementation does not take care of pasting the translated text exactly in place of the original English text, however i have plans to do that and more in my next iterations. 



## How You Can Access It
My HuggingFace - Gradio app can be accessed on my account page on thier website, its accessible to public and is available over here - [Translate English Text to Your Regional Language In Your Forwarded Images](https://huggingface.co/spaces/ysharma/TranslateQuotesInImageForwards).
Providing the demo in form of an animation below.

![](/images/gif1.gif)
  
 
## Conclusion :
## Benefits
[HuggingFace Spaces](https://huggingface.co/spaces) is a cool new feature, where anyone can host their AI models using two awesome SDKs - either [Streamlit](https://streamlit.io/) and Gradio. Spaces are a simple way to host ML demo apps directly on your HF profile page or your organization’s profile. This empowers our ML community to create our own little ML project portfolios, showcasing our projects at conferences, to stakeholders, or to any interested parties and to work collaboratively with other people in the ecosystem.

## Ease of use
Few points to keep in mind for an easy passage while building a complex Gradio app like this one -

* All the required libraries should be mentioned in *requirements.txt* file 
* In case you have some *Debian dependencies* and you would want to use sudo apt install for the same, make sure you copy such libraries in *packages.txt* file
* Make sure you are copying all the supporting files (images/fonts) over to your *app space repo*
* Comment aptly the code that you are submiiting under *app.py* file  
* Try to have your model and tokenizers loading outside the inference calls made from gradio.interface(). This helps in speeding up your app response to the users.
* This apps *app.py* code can help you take an inspiration in case you want to have multiple and different type of inputs and outputs (image/text/radio box *etc.*). It took me a while to figure out the right way. 

## Awesome Community
At the end of the day a strong community support helps you in learning about cool new avenues, uderstanding hard concepts, in resolving your issues, and in staying motivated to improve yourself and your skills. Among many incredible folks out there building for community, I would like to take a moment and thanks a few of them for all the efforts they put in. Reach out to them on Twitter and Discord here -- 

* [Abubakar Abid](https://twitter.com/abidlabs), [Ali](https://twitter.com/si3luwa),[AK](https://twitter.com/ak92501) of Gradio labs
* [Merve Noyan](https://twitter.com/mervenoyann), [Omar](https://twitter.com/osanseviero) from HuggingFace
* [HuggingFace Discord community](http://hf.co/join/discord)

## Future work
The app is still a bit rough on the edges and I plan to improve it in future interations, for example, right now it might not process well certain screenshots and those images in which the text is slanted a bit. Planning to enable OCR for slant text and for images in which text is present at multiple places. I will also be adding more languages to the mix. And lastly would be trying to insert the translated text at the same spot as the original image and in similar font style and font size.

## Lastmile AI
Gradio helps in bridging the gap between developing your ML models and showcasing them to the world. In my humble opinion, this is a crucial step in two main themes of this new year - Democratizing AI and Productioninzing AI.  


My github repo and code can be accessed over here - [HugginFace_Gradio](https://github.com/yvrjsharma/HugginFace_Gradio/blob/main/Whatsapp_Image_Forwards_In_Your_Language_GradioDemo.ipynb).

*If you enjoyed this article, please feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/yuvraj-sharma-a7154628/) or [Twitter](https://twitter.com/yvrjsharma) and do share your feedback and any other ML app ideas that you would want to implement yourself, I will be happy to help as much as I can*


Image source - Photo by <a href="https://unsplash.com/@rev3n?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Michał Kubalczyk</a> on <a href="https://unsplash.com/s/photos/tech?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
