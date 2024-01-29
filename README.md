# Youtube-Summarizer

Welcome to the Youtube Summarizer. This summarizes youtube videos using NLP.

When using extractive summarization, the algorithm will take the relevant passage and group its most essential paragraphs together to produce the summary text.

When using abstractive summarization, the algorithm will use its own words to construct a summary based on the provided passage. Compared to extractive summarization, this is more complicated.

We are utilizing extractive summarization for our YouTube summarizer. We use a variety of summary approaches for extractive summarization, such as TFIDF Vectorizer, BART, or Bidirectional and Auto-Regressive Transformer.

Term frequency-inverse document frequency, or TF-IDF, is a vectorizer that turns text into a vector format. There are two terms in it: inverse document frequency and term frequency. The product of these two components is the TF-IDF value. The number of times a word appears in a sentence divided by the total number of words in the sentence is known as term frequency. The log of the number of sentences by the number of sentences that contain the supplied word is known as the inverse document frequency. Okay, let's get the implementation going.


The following is an example:
1 - Introduction to Generative AI by Google Cloud Tech(https://www.youtube.com/watch?v=G2fqAlgmoPo) - credits

The actual script:
GWENDOLYN STRIPLING: Hello. And welcome to Introductionto Generative AI. My name is Dr.Gwendolyn Stripling. And I am theartificial intelligence technical curriculum developerhere at Google Cloud. In this course, you learnto define generative AI, explain how generative AI works,describe generative AI model types, and describegenerative AI applications. Generative AI is a typeof artificial intelligence technology that can producevarious types of content, including text, imagery,audio, and synthetic data. But what is artificialintelligence? Well, since we aregoing to explore generative artificialintelligence, let's provide a bit of context. So two very commonquestions asked are what is artificialintelligence and what is the differencebetween AI and machine learning. One way to think about itis that AI is a discipline, like physics for example. AI is a branch ofcomputer science that deals with the creationof intelligence agents, which are systems that can reason,and learn, and act autonomously. Essentially, AI has to dowith the theory and methods to build machines thatthink and act like humans. In this discipline, wehave machine learning, which is a subfield of AI. It is a program or system thattrains a model from input data. That trained model canmake useful predictions from new or neverbefore seen data drawn from the same oneused to train the model. Machine learninggives the computer the ability to learn withoutexplicit programming. Two of the most common classesof machine learning models are unsupervised andsupervised ML models. The key differencebetween the two is that, with supervisedmodels, we have labels. Labeled data 
is data that comeswith a tag like a name, a type, or a number. Unlabeled data is datathat comes with no tag. This graph is anexample of the problem that a supervised modelmight try to solve. For example, let's say youare the owner of a restaurant. You have historicaldata of the bill amount and how much different peopletipped based on order type and whether it waspicked up or delivered. In supervised learning, themodel learns from past examples to predict future values,in this case tips. So 
here the model usesthe total bill amount to predict the future tip amountbased on whether an order was picked up or delivered. This is an exampleof the problem that 
an unsupervisedmodel might try to solve. So here you want to lookat tenure and income and then group orcluster employees to see whether someoneis on the fast track. 
Unsupervised problemsare all about discovery, about looking at the raw dataand seeing if it naturally falls into groups. Let's get a little deeperand show this graphically as understandingthese concepts are the foundation for yourunderstanding of generative AI. In supervised learning,testing data values or x are input into the model. The model outputs a predictionand compares that prediction to the training dataused to train the model. If the predicted test datavalues and actual training data values are far apart,that's called error. And the model triesto reduce this error until the predicted and actualvalues are closer together. This is a classicoptimization problem. Now that we'veexplored the difference between artificial intelligenceand machine learning, and supervised andunsupervised learning, let's briefly explorewhere deep learning fits as a subset ofmachine learning methods. While machine learningis a broad field that encompasses manydifferent techniques, deep learning is a typeof machine learning that uses artificialneural networks, allowing them to process morecomplex patterns than machine learning. Artificial neural networks areinspired by the human brain. They are made up of manyinterconnected nodes or neurons that can learn to perform tasksby processing data and making predictions. Deep 
learning modelstypically have many layers of neurons, whichallows them to learn more complex patterns thantraditional machine learning models. And neural networks can useboth labeled and unlabeled data. This is calledsemi-supervised learning. In semi-supervisedlearning, a neural network is trained on a smallamount of labeled data and a large amountof unlabeled data. The labeled data helpsthe neural network to learn the basicconcepts of the task while the unlabeled datahelps the neural network to generalize to new examples. Now we finally get towhere generative AI fits into this AI discipline. Gen AI is a subset ofdeep learning, which means it uses artificialneural networks, can process both labeledand unlabeled data using supervised, unsupervised,and semi-supervised methods. Large language models are alsoa subset 
of deep learning. Deep learning models, or machinelearning models in general, can be divided into two types,generative and discriminative. A discriminative modelis a type of model that is used to classify orpredict labels for data points. Discriminativemodels are typically trained on a data setof labeled data points. And they learn the relationshipbetween the features of the data pointsand the labels. Once a discriminativemodel is trained, it can be used to predict thelabel for new data points. A generative modelgenerates new data instances based on a learned probabilitydistribution of existing data. Thus generative modelsgenerate new content. Take this example here. The discriminative model learnsthe conditional probability distribution or theprobability of y, our output, given x, ourinput, that this is a dog and classifies it asa dog and not a cat. The generative model learns thejoint probability distribution or the probability ofx and y and predicts the conditional probabilitythat this is a dog and can then generatea picture of a dog. So to summarize,generative models can generate new data instanceswhile discriminative models discriminate between differentkinds of data instances. The top image showsa traditional machine learning model whichattempts to learn the relationship betweenthe data and the label, or what you want to predict. The bottom image showsa generative AI model which attempts to learnpatterns on content so that it can generate new content. A good way to distinguishwhat is gen AI and what is not is shown in this illustration. It is not gen AI when theoutput, or y, or label is a number or a class, forexample spam or not spam, or a probability. It is gen AI when the output isnatural language, like speech or text, an image oraudio, for example. Visualizing this mathematicallywould look like this. If you haven't seenthis for a while, the y is equal to f ofx equation calculates the dependent output of aprocess given different inputs. 
The y stands forthe model output. The f embodies the functionused in the calculation. And the x represents the inputor inputs used for the formula. So the model output is afunction of all the inputs. If the y is the number,like predicted sales, it is not gen AI. If y is a sentence,like define sales, it is generative as the questionwould elicit a text response. The response would be basedon all the massive large data the model wasalready trained on. To summarize at a high level,the traditional, classical supervised and unsupervisedlearning process takes training code andlabel data to build a model. Depending on theuse case or problem, the model can giveyou a prediction. It can classify somethingor cluster something. We use this example to showyou how much more robust the gen AI process is. The gen AI process can taketraining code, label data, and unlabeled dataof all data types and build a foundation model. The foundation model canthen generate new content. For example, text, 
code,images, audio, video, et cetera. We've come a long away fromtraditional programming to neural networksto generative models. In traditionalprogramming, we used to have to hard code the rulesfor distinguishing a cat-- the type, animal; legs,four; ears, two; fur, yes; likes yarn and catnip. In the wave ofneural networks, we could give the networkpictures of cats and dogs and ask is this a cat andit would predict a cat. In the generativewave, we as users can generate our owncontent, whether it be text, images, audio,video, et cetera, for example models like PaLM orPathways Language Model, or LAMBDA, Language Modelfor Dialogue Applications, ingest very, very large datafrom the multiple sources across the internet andbuild foundation language models we can use simplyby asking a question, whether typing it intoa prompt or verbally talking into the prompt itself. So when you ask itwhat's a cat, it can give you everything ithas learned about a cat. Now we come to ourformal definition. What is generative AI? Gen AI is a type ofartificial intelligence that creates new contentbased on what it has learned from existing content. The process of learningfrom existing content is called training andresults in the creation of a statistical modelwhen given a prompt. AI uses the model to predictwhat an expected response might be and this generatesnew content. Essentially, it learnsthe underlying structure of the data andcan then generate new samples that are similarto the data it was trained on. As previously mentioned, agenerative language model can take what it has learnedfrom the examples it's been shown and createsomething entirely new based on that information. Large language models areone type of generative AI since they generate novelcombinations of text in the form of naturalsounding language. A generative imagemodel takes an image as input and can output text,another image, or video. For example, underthe output text, you can get visualquestion answering while under output image, animage completion is generated. And under output video,animation is generated. A generative languagemodel takes text as input and can output more text, animage, audio, or decisions. For example, underthe output text, question answering is generated. And under output image,a video is generated. We've 
stated that generativelanguage models learn about patterns and languagethrough training data, then, given some text, theypredict what comes next. Thus generative language modelsare pattern matching systems. They learn about patternsbased on the data you provide. Here is an example. Based on things it's learnedfrom its training data, it offers predictions of howto complete this sentence, I'm making a sandwich withpeanut butter and jelly. Here is the sameexample using Bard, which is trained on amassive amount of text data and is able tocommunicate and generate humanlike text in responseto a wide range of prompts and questions. Here is another example. The meaning of life is-- and Bart gives youa contextual answer and then shows the highestprobability response. The power of generative AI comesfrom the use of transformers. Transformers produceda 2018 revolution in natural language processing. At a high level, atransformer model consists of anencoder and decoder. The encoder encodesthe input sequence and passes it tothe decoder, which learns how to decodethe representation for a relevant task. In transformers, hallucinationsare words or phrases that are generatedby the model that are often nonsensical orgrammatically incorrect. Hallucinations can be causedby a number of factors, including the model is nottrained on enough data, or the model is trainedon noisy or dirty data, or the model is notgiven enough context, or the model is notgiven enough constraints. Hallucinations can be aproblem for transformers because they can make the outputtext difficult to understand. They can also makethe model more likely to generate incorrector misleading information. A prompt is ashort piece of text that is given to the largelanguage model as input. And it can be used to controlthe output of the model in a variety of ways. Prompt design is theprocess of creating a prompt that willgenerate the desired output from a large language model. As previously mentioned,gen AI depends a lot on the training data thatyou have fed into it. And it analyzes the patternsand structures of the input data and thus learns. But with access to a browserbased prompt, you, the user, can generate your own content. We've shown illustrations of thetypes of input based upon data. Here are theassociated model types. Text-to-text. Text-to-text models takea natural language input and produces a text output. These models are trainedto learn the mapping between a pair of text, e.g. for example, translationfrom one language to another. Text-to-image. Text-to-image models are trainedon a large set of images, each captioned with ashort text description. Diffusion is one methodused to achieve this. Text-to-video and text-to-3D. Text-to-video models aim togenerate a video representation from text input. The input text can be anythingfrom a single sentence to a full script. And the output is a video thatcorresponds to the input text. Similarly, text-to-3Dmodels generate three dimensional objects thatcorrespond to a user's text description. For example, this can be usedin games or other 3D worlds. Text-to-task. Text-to-task models are trainedto perform a defined task or action based on text input. This task can be awide range of actions such as answering a question,performing a search, making a prediction, ortaking some sort of action. For example, atext-to-task model could be trained to navigate aweb UI or make changes to a doc through the GUI. A foundation model is alarge AI model pre-trained on a vast quantity of datadesigned to be adapted or fine tuned to a wide rangeof downstream tasks, such as sentiment analysis,image captioning, and object recognition. Foundation modelshave the potential to revolutionize manyindustries, including health care, finance,and customer service. They can be used todetect fraud and provide personalized customer support. Vertex AI offers amodel garden that includes foundation models. The language foundationmodels include PaLM API for chat and text. The vision foundation modelsincludes stable diffusion, which has been shown tobe effective at generating high quality imagesfrom text descriptions. Let's say you havea use case where you need to gather sentimentsabout how your customers are feeling about yourproduct or service. You can use the classificationtask sentiment analysis task model for just that purpose. And what if you needed toperform occupancy analytics? There is a task modelfor your use case. 
Shown here are genAI applications. Let's look at an exampleof code generation shown in the second blockunder code at the top. In this example, I've input acode file 
conversion problem, converting from Python to JSON. I use Bard. And I insert into theprompt box the following. I have a Pandas DataFrame withtwo columns, one with the file name and one with the hourin which it is generated. I'm trying to convertthis into a JSON file in the format shown onscreen. Bard returns the steps I needto do this and the code snippet. And here my outputis in a JSON format. It gets better. I happen to be using Google'sfree, browser-based Jupyter Notebook, known as Colab. And I simply export thePython code to Google's Colab. To summarize, Bartcode generation can help you debug yourlines of source code, explain your codeto you line by line, craft SQL queriesfor your database, translate code from onelanguage to another, and generate documentationand tutorials for source code. Generative AI Studio lets youquickly explore and customize gen AI models that you canleverage in your applications on Google Cloud. Generative AI Studio helpsdevelopers create and deploy Gen AI models by providing avariety of tools and resources that make it easyto get started. For example, there's alibrary of pre-trained models. There is a tool forfine tuning models. There is a tool for deployingmodels to production. And there is a communityforum for developers to share ideas and collaborate. Generative AI AppBuilder lets you create gen AI apps withouthaving to write any code. Gen AI App Builder has adrag and drop interface that makes it easy todesign and build apps. It 
has a visualeditor that makes it easy to createand edit app content. It has a built-insearch engine that allows users to search forinformation within the app. And it has aconversational AI Engine that helps users tointeract with the app using natural language. You can create your own digitalassistants, custom search engines, knowledge bases,training applications, and much more. PaLM API lets youtest and experiment with Google's large languagemodels and gen AI tools. To make prototyping quickand more accessible, developers can integratePaLM API with Maker suite and use it to access theAPI using a graphical user interface. The suite includes a number ofdifferent tools such as a model training tool, a modeldeployment tool, and a model monitoring tool. The model training tool helpsdevelopers train ML models on their data usingdifferent algorithms. The model deployment tool helpsdevelopers deploy ML models to production with a number ofdifferent deployment options. The model monitoringtool helps developers monitor the performanceof their ML models in production using adashboard and a number of different metrics. Thank you for watchingour course, Introduction to Generative AI.
Truncation was not explicitly activated but `max_length` is provided a specific value, please use `truncation=True` to explicitly truncate examples to max length. Defaulting to 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy more precisely by providing a specific strategy to `truncation`.

YouTube video Summary:
Generative AI is a type of artificial intelligence technology that can producevarious types of content, including text, imagery,audio, and synthetic data. In this course, you learn how generative AI works, explain how it works,describe generativeAI model types, and describe Generative AI applications. Dr.Gwendolyn Stripling is theartificial intelligence technical curriculum developer at Google Cloud. She explains the difference between AI and machine learning. She also explains how 
to use generative artificialintelligence in your own life. She ends the course with a question and answer session on how to apply it to your life.