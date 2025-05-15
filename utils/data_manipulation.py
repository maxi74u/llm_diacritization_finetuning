import pyarabic.araby
from typing import List,Optional
def construct_input_out_pair_batch(
    batch, 
    system_prompt: Optional[str] = None, 
    user_command: Optional[str] = '', 
    diacritized_col_name: str = 'text'
):
    # Extract lists of diacritized strings
    diacritized_list = batch[diacritized_col_name]
    # Strip tashkeel for each
    undiacritized_list = [pyarabic.araby.strip_tashkeel(s) for s in diacritized_list]

    all_messages = []
    for und in undiacritized_list:
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user",      "content": f"{user_command} {und}"})
        msgs.append({"role": "assistant", "content": und})
        all_messages.append(msgs)

    # Return a dict whose values are lists (one per example in the batch)
    return {"messages": all_messages}

def construct_input_out_pairs(
    diacritized_ds,
    system_prompt: Optional[str] = None,
    keep_columns: List[str] = [],
    user_command: str = '',
    diacritized_col_name: str = 'text',
    batch_size: int = 1000,      # you can tune this
):
    # figure out which columns to drop
    to_remove = [c for c in diacritized_ds.column_names if c not in keep_columns]

    return diacritized_ds.map(
        lambda batch: construct_input_out_pair_batch(
            batch,
            system_prompt=system_prompt,
            user_command=user_command,
            diacritized_col_name=diacritized_col_name
        ),
        batched=True,
        batch_size=batch_size,
        remove_columns=to_remove,
    )
