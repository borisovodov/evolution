import random
import torch
import torch.nn as nn
import torch.nn.functional as functional
from typing import Self, Optional

INPUT_SIZE = 34
OUTPUT_SIZE = 12
MUTATION_RATE = 0.01


class Genes():
    def __init__(self, father_genes: Optional[Self], mother_genes: Optional[Self]):
        if father_genes and mother_genes:
            self.model = TransformationModel.from_crossover(father_genes=father_genes, mother_genes=mother_genes)
            self.model.mutation()
        else:
            self.model = TransformationModel()
    
    def decide(
        self,
        north_neighbour: float,
        north_neighbour_satiety: float,
        north_neighbour_age: float,
        north_neighbour_similarity: float,
        north_east_neighbour: float,
        north_east_neighbour_satiety: float,
        north_east_neighbour_age: float,
        north_east_neighbour_similarity: float,
        east_neighbour: float,
        east_neighbour_satiety: float,
        east_neighbour_age: float,
        east_neighbour_similarity: float,
        south_east_neighbour: float,
        south_east_neighbour_satiety: float,
        south_east_neighbour_age: float,
        south_east_neighbour_similarity: float,
        south_neighbour: float,
        south_neighbour_satiety: float,
        south_neighbour_age: float,
        south_neighbour_similarity: float,
        south_west_neighbour: float,
        south_west_neighbour_satiety: float,
        south_west_neighbour_age: float,
        south_west_neighbour_similarity: float,
        west_neighbour: float,
        west_neighbour_satiety: float,
        west_neighbour_age: float,
        west_neighbour_similarity: float,
        north_west_neighbour: float,
        north_west_neighbour_satiety: float,
        north_west_neighbour_age: float,
        north_west_neighbour_similarity: float,
        satiety: float,
        age: float
    ) -> tuple[
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
        #if self.model == TransformationModel.first_model:
        #    print(f"""input: {[
        #        north_neighbour,
        #        north_neighbour_satiety,
        #        north_neighbour_age,
        #        north_neighbour_similarity,
        #        north_east_neighbour,
        #        north_east_neighbour_satiety,
        #        north_east_neighbour_age,
        #        north_east_neighbour_similarity,
        #        east_neighbour,
        #        east_neighbour_satiety,
        #        east_neighbour_age,
        #        east_neighbour_similarity,
        #        south_east_neighbour,
        #        south_east_neighbour_satiety,
        #        south_east_neighbour_age,
        #        south_east_neighbour_similarity,
        #        south_neighbour,
        #        south_neighbour_satiety,
        #        south_neighbour_age,
        #        south_neighbour_similarity,
        #        south_west_neighbour,
        #        south_west_neighbour_satiety,
        #        south_west_neighbour_age,
        #        south_west_neighbour_similarity,
        #        west_neighbour,
        #        west_neighbour_satiety,
        #        west_neighbour_age,
        #        west_neighbour_similarity,
        #        north_west_neighbour,
        #        north_west_neighbour_satiety,
        #        north_west_neighbour_age,
        #        north_west_neighbour_similarity,
        #        satiety,
        #        age
        #    ]}""")
        output = self.model(torch.tensor([
            north_neighbour,
            north_neighbour_satiety,
            north_neighbour_age,
            north_neighbour_similarity,
            north_east_neighbour,
            north_east_neighbour_satiety,
            north_east_neighbour_age,
            north_east_neighbour_similarity,
            east_neighbour,
            east_neighbour_satiety,
            east_neighbour_age,
            east_neighbour_similarity,
            south_east_neighbour,
            south_east_neighbour_satiety,
            south_east_neighbour_age,
            south_east_neighbour_similarity,
            south_neighbour,
            south_neighbour_satiety,
            south_neighbour_age,
            south_neighbour_similarity,
            south_west_neighbour,
            south_west_neighbour_satiety,
            south_west_neighbour_age,
            south_west_neighbour_similarity,
            west_neighbour,
            west_neighbour_satiety,
            west_neighbour_age,
            west_neighbour_similarity,
            north_west_neighbour,
            north_west_neighbour_satiety,
            north_west_neighbour_age,
            north_west_neighbour_similarity,
            satiety,
            age
        ])).tolist()
        #if self.model == TransformationModel.first_model:
        #    print(f"output: {output}")
        return output


class TransformationModel(nn.Module):
    hidden_size = 40
    #first_model: Optional[Self] = None

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size, OUTPUT_SIZE)
        #if self.__class__.first_model == None:
        #    self.__class__.first_model = self

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
        father_model = father_genes.model
        mother_model = mother_genes.model
        child_model = TransformationModel()
        for child_param, father_param, mother_param in zip(child_model.parameters(), father_model.parameters(), mother_model.parameters()):
            child_param.data = (father_param.data + mother_param.data) / 2
        return child_model
