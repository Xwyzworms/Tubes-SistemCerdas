from typing import Any, Dict, List, Optional
import numpy as np

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

def linReg(inputs : np.array , outputs : np.array) -> Dict[str,Any]:
    """
    Calculate SST , SSR ,RSquared and SSE
    SST : Total Variability
    SSR : How Well Line fits the data
    RSquared : How Well Line Explained the data
    Args:
        inputs (np.array): [description]
        outputs (np.array): [description]

    Returns:
        Dict: RSquared, Coeff,and SSE
    """
    info : Dict[str,Any] = {}
    X : np.array = inputs.copy()
    Y : np.array = outputs.copy()
    
    coeff = np.dot( np.linalg.pinv(np.dot(X.transpose(), X)), np.dot(X.transpose(), Y) ) 
    Y_mean : np.float = np.mean(Y)
    yPred : np.array = np.dot(X, coeff)
    
    SST : np.float = np.sum(np.array([(i - Y_mean) ** 2 for i in Y]),axis=None)
    SSR : np.float = np.sum(np.array([(y - ypred)**2 for y,ypred in zip(Y,yPred)]),axis= None)
    Rsquared : np.float = 1 - (SSR / SST) * 100
    SSE = (SSR / len(Y))

    info["Rsquared"] = Rsquared
    info["coeff" ] =  coeff
    info["error"] = SSE
    
    return info

def terminate_GA(best : Dict[str,Any]) -> bool:
    """
     Terminate the Genetic Algorithm if the RSquared > 98.0

    Args:
        best (Dict[str,Any]): dict Of individual Information

    Returns:
      bool  
    """
    if best["Rsquared"] > 98.0:
        return True
    return False


class Population:
    
    def __init__(self,TotalSize,TotalIndividu : int) -> None:
        self.Totalgenome: int = TotalIndividu
        self.totalSize : int =TotalSize

    def createIndividu(self) -> np.array:
        """
        Create Individu using Binary 

        Returns:
            List[int]: binary sequence [0,1,0,1 ....] 
        """
        return np.random.choice([0,1],size=self.Totalgenome)
    
    def createPopulation(self) -> List[np.array] :
        """
        Create Population with x size

        Returns:
            List: list of individu
        """
        return [self.createIndividu() for _ in range(self.totalSize)]
   
    def fitness(self, individual : List[int], inputs : np.array, yTrue : np.array) -> Dict[str,Any]:
        """
        
        Fitness current Individual

        Args:
            individual (List[int]): [description]
            inputs (np.array): [description]

        Returns:
            Dict[str,Any]: [description]
        """
        info : Dict[str,Any] = {}
        predicted : np.array = np.dot(np.array(individual), inputs)
        yTrue_mean : np.float = np.mean(yTrue) 

        SST : np.float = np.sum(np.array([(y - yTrue_mean) ** 2 for y in yTrue]),axis=None)
        SSR : np.float = np.sum(np.array([(ytrue - ypred) ** 2 for ytrue,ypred in zip(yTrue,predicted)]),axis = None)
        Rsquared : np.float = 1 - (SSR / SST) * 100

        SSE : np.float = SSR / len(yTrue)

        info["Rsquared"] = Rsquared
        info["coeff"] = individual
        info["error"] = SSE

        return info
if __name__ == "__main__":
    pop= Population(10,5)
    print(pop.createPopulation())