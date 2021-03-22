from typing import List
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

def linReg(inputs : np.array , outputs : np.array):
    """
    Calculate SST , SSR ,COD To determine wheter the fitted line explained the data


    Args:
        inputs (np.array): [description]
        outputs (np.array): [description]

    Returns:
        [type]: [description]
    """

    X : np.array = inputs.copy()
    Y : np.array = outputs.copy()
    #coeff = np.dot( np.linalg.pinv(X), np.dot(X.transpose(), Y))    
    #print(coeff)
    #Y_mean : np.array = np.mean(Y)
    #SST : np.array = np.sum(np.array([(i - Y_mean) ** 2 for i in Y]))


class Population:
    
    def __init__(self,TotalSize,TotalIndividu : int) -> None:
        self.Totalgenome: int = TotalIndividu
        self.genomes = List[int]
        self.totalSize : int =TotalSize

    def createIndividu(self) -> List[int]:
        """
        Create lists of individu
        """
        return [np.random.choice([0,1],size=self.Totalgenome) for _ in range(self.totalSize)]


if __name__ == "__main__":
    x,y = generate_data(100)
    print(x)
    linReg(x,y)