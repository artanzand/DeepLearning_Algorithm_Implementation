# Deep Learning - Algorithm Implementation


This repository contains the implementation of deep learning algorithms from scratch. The goal is to have a single resource where people can find all kinds of possible implementations of basic algorithms in DL so that this becomes a standard reference for base models and projects involving the use of these algorithms.  

My intention in this repo is to build deep Neural Networks in three gradual steps. I will be walking through the steps and math in ipynb files.

1. One-node Neural Network
> To simulate a Logistic Regression through a neural net with just one node.

2. Neural Network with one hidden layer 
> To build a complete 2-class classification neural network with a hidden layer. No regularization implemented. tanh() will be the activaiton function for the hidden layer and sigmoid() will be the activation function for the output layer.

3. Deep Neural Network
> To implement all the building blocks of a neural network and use the building blocks in the previous part to build a neural network of any architecture. 
> No regularization will be implemented at this stage.  


### Future Improvement
This implementation of neural network from scratch was just a demonstration of how we could implement the model using the underlying math. The next improvement could be adding regularization to the model. However, the proper way of designing a model is to include them in a Class function to allow for attributes like fit and predict, and to have access to the calculated weights and biases. This could be follow up project to this model development
<br></br>

## CONTRIBUTING

### Follow the steps below to contribute:
1. Fork the repository.
2. Add the implementation of the algorithm with a clearly defined filename for the script or the notebook.
3. Test the implementation thoroughly and make sure that it works with some dataset.
4. Add a link with a short description about the file in the [README.md](https://github.com/adityashrm21/Deep-Learning-Algorithms-Implementation/blob/master/README.md).
5. Create a pull request for review with a short description of your changes.
6. Do not forget to add attribution for references and sources used in the implementation.

Sources:
- [DEEPLEARNING.AI](https://www.deeplearning.ai/)
