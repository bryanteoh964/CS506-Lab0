## Please fill in all the parts labeled as ### YOUR CODE HERE

import numpy as np
import pytest
from utils import *

def test_dot_product():
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])
    
    result = dot_product(vector1, vector2)
    
    assert result == 32, f"Expected 32, but got {result}"
    
def test_cosine_similarity():
    vector1 = np.array([1, 0, 0])
    vector2 = np.array([0, 1, 0])
    
    result = cosine_similarity(vector1, vector2)
    
    expected_result = 0.0  # Cosine similarity of orthogonal vectors should be 0
    
    assert np.isclose(result, expected_result), f"Expected {expected_result}, but got {result}"

def test_nearest_neighbor():
    target_vector = np.array([1, 0, 0])
    vectors = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    
    result = nearest_neighbor(target_vector, vectors)
    
    expected_index = 1  # The second vector is identical to the target_vector
    
    assert result == expected_index, f"Expected index {expected_index}, but got {result}"
