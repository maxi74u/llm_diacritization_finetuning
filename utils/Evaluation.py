from pyarabic.araby import strip_tashkeel
import re
import pandas as pd

from pyarabic import araby
from prettytable import PrettyTable
import re
from tqdm import tqdm
import warnings
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from pyarabic.araby import LETTERS
from datasets import load_dataset
from tqdm import tqdm
from .Sadeed_Evaluation import ArabicDiacritizationEvaluator
from .alignments import alignment

def evaluation(huggingface_dataset_name:str, split:str, X_col_name:str, Y_col_name:str, Y_pred_col_name:str, token_huggingface:str=None, verbose = False)->str:
    """
    Evaluate the performance of a model on a given dataset.

    Args:
        huggingface_dataset_name (str): The name of the Hugging Face dataset to evaluate.
        split (str): The split of the dataset to evaluate ('train', 'validation', 'test').
        X_col_name (str): The name of the column containing input data.
        Y_col_name (str): The name of the column containing ground truth data.
        Y_pred_col_name (str): The name of the column containing predicted data.
        token_huggingface (str): Optional token for Hugging Face authentication, if you want to push the new dataset after alignments to Hugging Face in the same name.
        verbose (bool): Optional enables or disables printing.
    Returns:
        dataset: The evaluated dataset with aligned predictions.
        Total_WER (float): The total word error rate.
        Morph_WER (float): The morphological word error rate.
        Total_DER (float): The total diacritic error rate.
        Morph_DER (float): The morphological diacritic error rate.
        NVW (float): The rate of not fully diacritic words.
    """
    if verbose: print("loading data")
    full_dataset_name = huggingface_dataset_name
    dataset_name = huggingface_dataset_name.split("/")[1]
    dataset = load_dataset(full_dataset_name)

    
    if verbose: print("alignment")
    df_testing = pd.DataFrame({
        "input": dataset[split][X_col_name],
        "inference": dataset[split][Y_pred_col_name]
    })

    # Apply the function row-wise
    predection_alignmented = df_testing.apply(lambda x: alignment(x['input'], x['inference']), axis=1)

    if verbose: print("evaluation")
    Total_WER, Morph_WER, Total_DER, Morph_DER, NVW  = ArabicDiacritizationEvaluator.report_error_on_senenteces(diacritized_sentences = predection_alignmented,
                              ground_truth_sentences = dataset[split][Y_col_name], gt_missing_diacritic_is_error = False, verbose = verbose)

    if f"{Y_pred_col_name}_after_alignment" in dataset[split].features:
      ds = dataset[split].remove_columns([f"{Y_pred_col_name}_after_alignment"])
    else: 
      ds = dataset[split]
    ds = ds.add_column(f"{Y_pred_col_name}_after_alignment", predection_alignmented)
    if token_huggingface:
      if verbose:  print("push to hub")
      ds.push_to_hub(dataset_name, token=token_huggingface)
    
    return ds, Total_WER, Morph_WER, Total_DER, Morph_DER, NVW
