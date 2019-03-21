# Training data augmentation for rumour detection using context-sensitive neural language model with large credibility corpus

## Rumour Data set

Following Rumour Dataset are used in our experiment.

## Data Collection

Data collection is performed to collect social-temporal data set on Twitter platform for rumour candidate tweets.

## Semantic Relatedness Computation

Semantic Relatedness computation is to locate various forms of rumours based on textual variations. 
ELMo based contextual-sensitive language model is employed to learn representation of tweets and 
pairwise cosine similarity are computed between reference rumour tweets and rumour candidate tweets.

## Baseline Classification Model

We evaluated the effectiveness of our augmented rumour data in a state-of-the-art classification model for the task of rumour detection. You can find modified source code in [Multitask4Veracity](https://github.com/soojihan/Multitask4Veracity)
