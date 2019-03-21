# Training data augmentation for rumour detection using context-sensitive neural language model with large credibility corpus

## Rumour Data set

Following Rumour Dataset are used in our experiment.

* [CrisisLexT26](https://github.com/sajao/CrisisLex/tree/master/data/CrisisLexT26): Unlabeled Boston Bombing dataset are obtained from CrisisLexT26 corpus.

* [Twitter event datasets (2012-2016)](https://figshare.com/articles/Twitter_event_datasets_2012-2016_/5100460) : This is the Tweet corpus that is  used as candidate tweets for data augmentation. 

## Data Collection

Data collection is performed to collect social-temporal data set on Twitter platform for rumour candidate tweets.

## Semantic Relatedness Computation

Semantic Relatedness computation is to locate various forms of rumours based on textual variations. 
ELMo based contextual-sensitive language model is employed to learn representation of tweets and 
pairwise cosine similarity are computed between reference rumour tweets and rumour candidate tweets.

## Baseline Classification Model

We evaluated the effectiveness of our augmented rumour data in a state-of-the-art classification model for the task of rumour detection. You can find modified source code in [Multitask4Veracity](https://github.com/soojihan/Multitask4Veracity)
