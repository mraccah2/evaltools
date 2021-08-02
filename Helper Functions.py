# @title Helper Functions
# Imports
import time
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from evaltools import airtable
atform = airtable.AirtableForm('appn7VdPRseSoMXEG','W1D1_T1','https://portal.neuromatchacademy.org/api/redirect/to/97e94a29-0b3a-4e16-9a8d-f6838a5bd83d')


def checkExercise1(A, B, C, D):
  """
  Helper function for checking exercise.

  Args:
    A: torch.Tensor
    B: torch.Tensor
    C: torch.Tensor
    D: torch.Tensor
  Returns:
    Nothing.
  """
  errors = []
  # TODO better errors and error handling
  if not torch.equal(A.to(int),torch.ones(20, 21).to(int)):
    errors.append(f"Got: {A} \n Expected: {torch.ones(20, 21)} (shape: {torch.ones(20, 21).shape})")
  if not np.array_equal( B.numpy(),np.vander([1, 2, 3], 4)):
    errors.append("B is not a tensor containing the elements of Z ")
  if C.shape != (20, 21):
    errors.append("C is not the correct shape ")
  if not torch.equal(D, torch.arange(4, 41, step=2)):
    errors.append("D does not contain the correct elements")

  if errors == []:
    print("All correct!")

  else:
    [print(e) for e in errors]


def timeFun(f, dim, iterations, device='cpu'):
  iterations = iterations
  t_total = 0
  for _ in range(iterations):
    start = time.time()
    f(dim, device)
    end = time.time()
    t_total += end - start

  if device == 'cpu':
    print(f"time taken for {iterations} iterations of {f.__name__}({dim}, {device}): {t_total:.5f}")
  else:
    print(f"time taken for {iterations} iterations of {f.__name__}({dim}, {device}): {t_total:.5f}")
