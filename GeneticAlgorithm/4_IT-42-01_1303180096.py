from typing import Any, Dict, List, Optional,Set, Tuple 
import numpy as np
from random import sample,random
from math import floor

def generate_data(size : int = 100,coefs :int = 4) -> np.array:
    """
    Generate Random data < Continnous >
    size = How many observations
    coefs = How many features
    Returns:
        np array (x), (y)
    """
    coeffs : List[float] = [0.4, 0.3 ,0.2, -0.1]
    x : List[np.array] = [[np.random.normal() for _ in range(len(coeffs))] for _ in range(size) ]
    y : List[np.array] = [np.dot(i,coeffs) for i in x]
    
    return np.array(x), np.array(y)



class Population:
    
    def __init__(self,TotalSize,Totalgenome: int) -> None:
        self.Totalgenome: int = Totalgenome
        self.totalSize : int = TotalSize
        self.bestIndividuals : List =[]
        self.population = []

    def createIndividu(self) -> np.array:
        """
        Create Individu using Real Numbers

        Returns:
            List[int]: Real sequence
        """
        return np.random.normal(size=self.Totalgenome)
    
    def createPopulation(self) -> None :
        """
        Create Population with x size

        Returns:
            List: None
        """
        self.population.append([self.createIndividu() for _ in range(self.totalSize)])
   
    def fitness(self, individual , inputs : np.array, yTrue : np.array) -> Dict[str,Any]:
        """
        
        Fitness current Individual

        Args:
            individual (List[int]): [description]
            inputs (np.array): [description]

        Returns:
            Dict[str,Any]: [description]
        """
        info : Dict[str,Any] = {}
        bias : np.float = 0
        predicted : np.array =  bias + np.dot(inputs,individual) 
        yTrue_mean : np.float = np.mean(yTrue) 

        SST : np.float = np.sum(np.array([(y - yTrue_mean) ** 2 for y in yTrue]),axis=None)
        SSR : np.float = np.sum(np.array([(ytrue - ypred) ** 2 for ytrue,ypred in zip(yTrue,predicted)]),axis = None)
    
        bias :np.float = np.mean(np.sqrt(np.array([(ytrue - ypred) ** 2 for ytrue,ypred in zip(yTrue,predicted)])))
        RMSE : np.float = np.sqrt(SSR/ len(y))
        Rsquared : np.float = (1 - (SSR / SST))

        SSE : np.float = SSR / len(y)
        oneDivError : np.float = 1 / SSE

        info["Rsquared"] = Rsquared
        info["coeff"] = individual
        info["error"] = SSE
        info["RMSE"] = RMSE 
        info["divByOne"] = oneDivError
        info["bias"] = bias 

        return info
    
    def evaluate_population(self, x,y,selectionSize : int) :
        """
        Function to evaluate the best individual from current population

        Args:
            pop (List[np.array]): List of Individuals 

        Returns:
            None
        """
        fitness_list= [self.fitness(individual,x,y) for individual in self.population[0]]
        error_list : List[Dict[str,Any]] = sorted(fitness_list,key=lambda i : i["error"])
        best_individuals = error_list[: selectionSize]
        self.bestIndividuals.clear()
        self.bestIndividuals.append(best_individuals)
        
        print(f"Error {best_individuals[0]['error']}\n RSquared {best_individuals[0]['Rsquared']}\n") 


    def mutate(self,individual : List[float], probabilityMutating : float) -> List[float]:
        """
        
        Ini Fungsinya lakuin mutasi , dah taulah mutasi mah

        Args:
            individual (List[float]): [description]
            probabilityMutating (float): [description]

        Returns:
            List[float]: [description]
        """
        indx : List[int] = [i for i in range(len(individual))]

        totalMutatedGens : int = int(probabilityMutating * len(individual))
        indx_toMutate : List[int] = sample(indx,k = totalMutatedGens)
        for ind in indx_toMutate:
            choice : np.int = np.random.choice([-1,1])
            gene_transform : float = choice*random()

            individual[ind] = individual[ind] + gene_transform
        return individual
        
    def crossover(self, parent1 : Dict[str,Any], parent2 : Dict[str,Any]):
        """
        Intinya mantap mantapan Ngasilin Anak , dah gitu konsepnya
        Args:
            parent1 (Dict[str,Any]): [description]
            parent2 (Dict[str,Any]): [description]
        """
        
        anak_haram : Dict[int,Any] = {}
        index : List[int] = [i for i in range( self.Totalgenome )]
        indexRandomize : List[int] = sample(index, floor(0.5 * self.Totalgenome))
        IndexNotInRandomize : List[int] = [i for i in index if i not in indexRandomize]

        getCromosomeFromParent1 : List[Any] = [[i,parent1['coeff'][i]] for i in indexRandomize]
        getCromosomeFromParent2 : List[Any] = [[i,parent2["coeff"][i]] for i in IndexNotInRandomize]
        
        anak_haram.update({key :value for (key,value) in getCromosomeFromParent1})
        anak_haram.update({key : value for(key,value) in getCromosomeFromParent2})
    
        return [anak_haram[i] for i in index]

    def create_new_generation(self,probabilityMutating : float,probility_gene_mutating:float) -> None:
        """
        Create new population using the best individuals

        Args:
            best_individuals (List[float]): [description]
        """
        pasangan_sah = [sample(self.bestIndividuals[0],2) for _ in range( self.totalSize)]
        crossOverered_parents = [self.crossover(pasangan[0],pasangan[1]) for pasangan in pasangan_sah]
        pasangan_sah_indx = [i for i in range(self.totalSize)]
        pasanganCalonMutasi = sample(pasangan_sah_indx,floor(probabilityMutating * self.totalSize))

        PasanganMutasi = [[i,self.mutate(crossOverered_parents[i],probility_gene_mutating)] for i in pasanganCalonMutasi]
        for anakMutasi in PasanganMutasi:
            crossOverered_parents[anakMutasi[0]] = anakMutasi[1]
        self.population.clear()
        self.population.append(crossOverered_parents)
        

if __name__ == "__main__":
    x,y = generate_data(100)
    np.random.seed(123) 
    pop =Population(100,4)
    pop.createPopulation()
    selectionSize = floor(0.1 * 100)
    max_generations = 50

    probability_Indiv = 0.1
    probaiblity_gene = 0.25

    terminate_ga = False
    for i in range(max_generations + 1) :
        print(f"generation : {i}")
        pop.evaluate_population(x,y,10)
        pop.create_new_generation(probability_Indiv,probaiblity_gene)
        
    
    print("Nilai x : ",x[0])
    print("Nilai Y : ",y[0])
    
    result = pop.bestIndividuals[0][0]['bias'] + np.dot(x[0],pop.bestIndividuals[0][0]["coeff"])
    print("Predicted = ",result)
    print(pop.bestIndividuals[0][0])

    #parent1 = pop.fitness(individual,x,y)
    #parent2 = pop.fitness(individual,x,y)