##############
Neural networks aka deep learning
##############

The first step to understanding neural networks is to be clear about the terms Artificial Intelligence
,Machine learning and Deep learning.**AI** includes practically any techinique that *empowers a machine to
take intelligent decisions without being expicitly programmed*.**Machine learning** resides within AI and
refers specifically to a range of tools that promote *intelligence through learning* .**Deep learning**
happens to be a part of machine learning that exploits *specific tools called* **Neural Networks** to make
it work.


Before Deep learning
=====================

Deep learning now constitutes most of machine learning and both advancements and econimic value generated are
growing exponentially ,but this was not always the case.Even in the last decade most of the ML community was
concerened with several now obsolete techniques such as kNN ,SVM and random forests.But these techinques had
several problems associated with them and nobody could get it to work really well on any problem that mattered
in real life.Let me demostrate them through two examples:

 - **Problem 1 :** Consider that the problem is to classify male and female voice.Going by old Ml techniques the first step would be to find out among the general properties of sound such as frequeny ,amplitude ,bass etc ,one that of differentiates them the most which would be frequency in this case.Such Properties of data are called features in ML.Now ,consider a database of voices of people speaking in different accents ,background noise etc.This causes a lot of confusion as to which features to select.Even in the male category voices of children might sometimes be interpreted as that of females due to the relatively high frequency.**We need to manually select the best features and this is a very hard ,ambiguous process.** The evolution of a separate field of study called **Feature Engineering** exemplifies this headache.

- **Problem 2 :** Consider that the problem is to recognize human faces.The traditional Ml method used for  this would be SVM[Support vector machine].It works be separating groups of input features.For instance ,one category might consist of with and without spectacles and another might be with sunglasses and without hair.As the complexity of the dataset increases,you can imagine that no of categories would increase and prediction accuracy would decrease exponentially.

Set up a GitHub account
=======================
