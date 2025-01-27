import random
import torch
import torch.nn as nn
import torch.nn.functional as functional
from typing import Self, Optional

INPUT_SIZE = 42
OUTPUT_SIZE = 12
MUTATION_RATE = 0.01


class Genes():
    def __init__(self, father_genes: Optional[Self], mother_genes: Optional[Self]):
        if father_genes and mother_genes:
            self.model = TransformationModel.from_crossover(father_genes=father_genes, mother_genes=mother_genes)
            self.model.mutation()
        else:
            self.model = TransformationModel()
    
    def decide(self, context: list[float]) -> list[
        float, # north
        float, # north-east
        float, # east
        float, # south-east
        float, # south
        float, # south-west
        float, # west
        float, # north-west
        float, # stay and defend
        float, # walk
        float, # love
        float # fight
    ]:
        output = self.model(torch.tensor(context)).tolist()
        return output


class TransformationModel(nn.Module):
    hidden_size = 50

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size, OUTPUT_SIZE)

    def forward(self, x):
        x = functional.relu(self.fc1(x)) # hidden layer activation
        x = functional.relu(self.fc2(x)) # hidden layer activation
        x = functional.relu(self.fc3(x)) # hidden layer activation
        return torch.sigmoid(self.fc4(x)) # output layer activation
    
    def mutation(self):
        for param in self.parameters():
            if random.random() < MUTATION_RATE:
                param.data += torch.randn(param.data.shape) * 0.1
    
    @staticmethod
    def from_crossover(father_genes: Genes, mother_genes: Genes) -> Self:
        #father_model = father_genes.model
        #mother_model = mother_genes.model
        #child_model = TransformationModel()
        #for child_param, father_param, mother_param in zip(child_model.parameters(), father_model.parameters(), mother_model.parameters()):
        #    child_param.data = (father_param.data + mother_param.data) / 2
        #return child_model
        return father_genes.model # TODO
