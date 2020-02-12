# Content Description

### Audio Feature Extraction
Code for extract the audio feature of one wav file using [openSMILE](https://www.audeering.com/opensmile/).

### Textual Features Extraction
In this file, we get an array of doc2Vec of the video transcription utterances. 
This array is then wrapped in a window of 50 words to later be used on a CNN to get the final textual features.
 
### Utterance Extraction with Praat TextGrid
With  a Praat TextGrid file split by silences, the video is segmented into utterances. 
It is generated the segmented video files, as well as the audio segmented files.
 
## MOSI

### Format Features
With a text and visual extracted features, generate a pickle with the format that the model presented [here](https://github.com/silvanabc/multimodal-sentiment-analysis) expects to receive.

### Text Extraction
A simple text extraction that generate a doc2vec array from the video utterances.
We used the pre-trained model of the English Wikipedia provided on [https://github.com/jhlau/doc2vec](https://github.com/jhlau/doc2vec).
