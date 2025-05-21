from pyarabic.araby import strip_tashkeel
import re
import pandas as pd

def alignment(
    raw_input: str,
    y_pred: str,
    reward: int = 1,
    penalty: int = -10,
    worst_case: float = -1e9,
    verbose: bool = False
) -> str:
    """
    Aligns a predicted Arabic string with a reference string based on a custom scoring system 
    that considers Arabic diacritics (Tashkeel). This function attempts to reconstruct the most 
    plausible prediction by aligning diacritic-laden characters to their undiacritized forms.

    Args:
        raw_input (str): The reference string.
        y_pred (str): The predicted string that will be aligned to the reference.
        reward (int, optional): Reward score for a correct character match. Defaults to 1.
        penalty (int, optional): Penalty score for a mismatch or insertion/deletion. Defaults to -10.
        worst_case (float, optional): Score for worst-case mismatch. Defaults to -1e9.
        verbose (bool, optional): If True, prints debugging info. Defaults to False.

    Returns:
        str: The aligned and diacritized output string based on the prediction.
    """    # define constant
    FATHA      = '\u064E'
    DAMMA      = '\u064F'
    KASRA      = '\u0650'
    TANWEEN_FATHA = '\u064B'  # ً
    TANWEEN_DAMMA = '\u064C'  # ٌ
    TANWEEN_KASRA = '\u064D'  # ٍ
    SUKUN      = '\u0652'
    SHADDA     = '\u0651'
    diacs = fr"[{FATHA}{DAMMA}{KASRA}{TANWEEN_FATHA}{TANWEEN_DAMMA}{TANWEEN_KASRA}{SUKUN}{SHADDA}]"

    # remove tashkeel from both strings
    raw_input_no_tashkeel , y_pred_no_tashkeel = strip_tashkeel(raw_input) , strip_tashkeel(y_pred)

    if (verbose):
        print(f"raw_input: {raw_input} -> {raw_input_no_tashkeel}")
        print(f"y_pred: {y_pred} -> {y_pred_no_tashkeel}")

    # define dp array
    dp = [[0 for _ in range(len(raw_input_no_tashkeel)+1)] for _ in range(len(y_pred_no_tashkeel)+1)]
    dp_path = [["" for _ in range(len(raw_input_no_tashkeel)+1)] for _ in range(len(y_pred_no_tashkeel)+1)]

    # set bondary value for dp array
    dp[0][0]=0
    dp_path[0][0]="done"
    for i in range(1,len(dp[0])):
        dp[0][i]=penalty
        dp_path[0][i]="left"

    for j in range(1,len(dp)):
        dp[j][0]=penalty
        dp_path[j][0]="up"


    # fill dp array
    it_y_pred, it_y_pred_no_tashkeel = 0, 0
    for it_dp_pred_i in range(1,len(dp)):
        while  it_y_pred < len(y_pred) :
            # if it is diacs or tanween increase it
            if re.match(diacs, y_pred[it_y_pred]):
                it_y_pred+=1
            else:
                break

        it_raw_input, it_raw_input_no_tashkeel = 0, 0
        for it_dp_ref_j in range(1,len(dp[it_dp_pred_i])):
            while  it_raw_input < len(raw_input) :
                # if it is diacs or tanween increase it
                if re.match(diacs, raw_input[it_raw_input]):
                    it_raw_input+=1
                else:
                    break
            ch1 = (dp[it_dp_pred_i-1][it_dp_ref_j-1] + reward) if y_pred_no_tashkeel[it_y_pred_no_tashkeel]==raw_input_no_tashkeel[it_raw_input_no_tashkeel] else worst_case
            ch2 = dp[it_dp_pred_i-1][it_dp_ref_j] + penalty
            ch3 = dp[it_dp_pred_i][it_dp_ref_j-1] + penalty
            dp[it_dp_pred_i][it_dp_ref_j], dp_path[it_dp_pred_i][it_dp_ref_j]=max((ch1,"diag"),(ch2,"up"),(ch3,"left"))
            it_raw_input+=1
            it_raw_input_no_tashkeel+=1

        it_y_pred+=1
        it_y_pred_no_tashkeel+=1


    # if (verbose):
    #     pd.DataFrame(dp_path, index = [" "]+list(y_pred_no_tashkeel), columns=[" "]+list(raw_input_no_tashkeel))


    # get the string
    i,j = len(dp_path)-1, len(dp_path[0])-1
    it_y_pred = len(y_pred)-1
    s=""
    while i>=0 and j>=0:
        if dp_path[i][j] == "diag":
            while  it_y_pred >= 0 :
                # if it is diacs  increase it
                if re.match(diacs, y_pred[it_y_pred]):
                    s+=y_pred[it_y_pred]
                    it_y_pred-=1
                else:
                    it_y_pred-=1
                    break
            s+=y_pred_no_tashkeel[i-1]
            i-=1
            j-=1
        elif (dp_path[i][j] == "left"):
            s+=raw_input_no_tashkeel[j-1]
            j=j-1
        elif dp_path[i][j] == "done":
            break
        else:
            while  it_y_pred >= 0 :
                if re.match(diacs, y_pred[it_y_pred]):
                    it_y_pred-=1
                else:
                    it_y_pred-=1
                    break
            i=i-1
    #reverse s
    s= s[::-1]

    return s
