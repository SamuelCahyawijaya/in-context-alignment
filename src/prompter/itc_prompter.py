###
# In-Context Learning with Alignment
###
def dict_list_to_list_dict(dict_data):
    return [dict(zip(dict_data,t)) for t in zip(*dict_data.values())]

###
# In-Context Table Completion
###
class ITCPrompter(object):
    __slots__ = ("headers", "header_template", "row_template", "query_template", "table_format", 
                 "query_keys", "target_keys", "label_keys", "source_keys")

    def __init__(self, headers: list[str], query_keys: list[str], label_keys: list[str],
                target_keys: list[str], source_keys: list[str], table_format: str = "markdown") -> None:
        self.headers = headers
        self.query_keys = query_keys
        self.target_keys = target_keys
        self.label_keys = label_keys
        self.source_keys = source_keys
        self.table_format = table_format
        
        if table_format == "markdown":
            header = f"| {' | '.join(self.headers)} |\n"
            header_separator = f"|{'|'.join('---' for _ in self.headers)}|\n"
            exemplar_template = ['{' + att_key + '}' for att_key in self.target_keys + self.label_keys + self.source_keys]
            query_template = ['{' + att_key + '}' for att_key in self.query_keys]
            
            self.header_template = header + header_separator
            self.row_template = f"| {' | '.join(exemplar_template)} |"
            self.query_template = f"| {' | '.join(query_template)} | [LABELS_CHOICE]"
        else:
            raise NotImplementedError()

    def generate_prompt(self, 
        input_exemplar: None | dict = None, 
        exemplars: None | dict[list] = None,
    ) -> str:
        # Init Rows
        rows = []

        # Format ICL
        if exemplars is not None:
            numel = len(exemplars[self.target_keys[0]])
            for i in range(numel):
                row_dict = {key: exemplars[key][i] for key in self.target_keys + self.label_keys + self.source_keys}
                rows.append(self.row_template.format(**row_dict))

        # Format Input Query
        sample_dict = {key: input_exemplar[key] for key in self.query_keys}
        rows.append(self.query_template.format(**sample_dict))

        # Return Resulting Prompt
        prompt = self.header_template + '\n'.join(rows)
        return prompt

if __name__ == '__main__':
    print('== TEST ITC ==')
    itc_prompter = ITCPrompter(
        headers=['TGT-SENT','LABEL','SRC-SENT'], query_keys=['text'],
        label_keys=['label'], target_keys=['text_1'], source_keys=['text_2']
    )
    
    print('ZERO-SHOT')
    print(
        itc_prompter.generate_prompt(
            input_exemplar={'text': 'Q1', 'label': 'XXX'}
        )
    )
    print()
    
    print('FEW-SHOT')
    print(
        itc_prompter.generate_prompt(
            input_exemplar={'text': 'Q1', 'label': 'XXX'},
            exemplars={
                'text_1': ['EXA1', 'EXA2', 'EXA3'], 
                'label': ['LBL1', 'LBL2', 'LBL3'], 
                'text_2': ['EXB1', 'EXB2', 'EXB3']
            }
        )
    )
    print()