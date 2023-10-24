###
# In-Context Learning with Alignment
###
def dict_list_to_list_dict(dict_data):
    return [dict(zip(dict_data,t)) for t in zip(*dict_data.values())]

class ICLPrompter(object):
    __slots__ = ("prompt_template", "icl_template", "iia_template")

    def __init__(self, prompt_template: str, icl_template: str | None = None, iia_template: str | None = None) -> None:
        self.prompt_template = prompt_template
        self.icl_template = icl_template
        self.iia_template = iia_template

        assert '{context}' in self.prompt_template
        assert '{query}' in self.prompt_template

    def generate_prompt(self, 
        input_exemplar: None | list[dict] | dict[list] = None, 
        icl_exemplars: None | list[dict] | dict[list] = None,
        input_alignment_exemplars: None | list[dict] | dict[list] = None,
        output_alignment_prompt: None | str = None
    ) -> str:
        # Init Prompt & Contexts
        prompt = self.prompt_template
        contexts = []

        # Format ICL
        if icl_exemplars is not None:
            if type(icl_exemplars) == dict:
                icl_exemplars = dict_list_to_list_dict(icl_exemplars)
                
            for ex in icl_exemplars:
                contexts.append(self.icl_template.format(**ex))

        # Format Input Alignment ICL
        if input_alignment_exemplars is not None:
            if type(input_alignment_exemplars) == dict:
                input_alignment_exemplars = dict_list_to_list_dict(input_alignment_exemplars)
                
            for ex in input_alignment_exemplars:
                contexts.append(self.iia_template.format(**ex))

        # Format Label Alignment ICL
        if output_alignment_prompt is not None:
            contexts.append(output_alignment_prompt)

        if len(contexts) > 0:
            context = '\n'.join(contexts)
        else:
            # Remove `[CONTEXT]`
            context = ''

        # Format Input Query
        input_exemplar['label'] = '[LABELS_CHOICE]' # Make label a variable for inference
        query = self.icl_template.format(**input_exemplar)
        prompt = prompt.format(**{'context': context, 'query': query})

        # Return Resulting Prompt
        return prompt

###
# In-Context Table Completion
###
class ITCPrompter(object):
    __slots__ = ("headers", "header_template", "row_template", "table_format", "attribute_keys")

    def __init__(self, headers: str, attribute_keys: str, table_format: str = "markdown") -> None:
        self.headers = headers
        self.attribute_keys = attribute_keys
        self.table_format = table_format
        
        if table_format == "markdown":
            header = f"| {' | '.join(self.headers)} |\n"
            header_separator = f"|{'|'.join('---' for _ in self.headers)}|\n"
            self.header_template = header + header_separator
            self.row_template = f"| {' | '.join('{' + att_key + '}' for att_key in self.attribute_keys)} |"
        else:
            raise NotImplementedError()

    def generate_prompt(self, 
        input_query: None | str = None, 
        exemplars: None | dict = None,
    ) -> str:
        # Init Rows
        rows = []

        # Format ICL
        if exemplars is not None:
            numel = len(exemplars[self.attribute_keys[0]])
            for i in range(numel):
                row_dict = {key: exemplars[key][i] for key in self.attribute_keys}
                rows.append(self.row_template.format(**row_dict))

        # Format Input Query
        rows.append(f'| {input_query} | [LABELS_CHOICE]')

        # Return Resulting Prompt
        prompt = self.header_template + '\n'.join(rows)
        return prompt
    
if __name__ == '__main__':
    icl_prompter = ICLPrompter(
        prompt_template="What is the sentiment of the following sentences?\n{context}\n{query}",
        icl_template="{input} => {label}",
        iia_template="{input} => {label}"
    )
    
    print('== TEST ICL ==')
    print('ZERO-SHOT')
    print(
        icl_prompter.generate_prompt(input_exemplar={'input': 'Q1', 'label': 'XXX'})
    )
    print()
    
    print('ICL Dict List')
    print(
        icl_prompter.generate_prompt(
            input_exemplar={'input': 'Q1', 'label': 'XXX'},
            icl_exemplars={
                'input': ['EX1', 'EX2', 'EX3'], 'label': ['LBL1', 'LBL2', 'LBL3']
            }
        )
    )    
    print()
    
    print('ICL List Dict')
    print(
        icl_prompter.generate_prompt(
            input_exemplar={'input': 'Q1', 'label': 'XXX'},
            icl_exemplars=[
                {'input': 'EX1', 'label': 'LBL1'},
                {'input': 'EX2', 'label': 'LBL2'},
                {'input': 'EX3', 'label': 'LBL3'}
            ]
        )
    )    
    print()

    print('INPUT-ALIGN')
    print(
        icl_prompter.generate_prompt(
            input_exemplar={'input': 'Q1', 'label': 'XXX'},
            input_alignment_exemplars={
                'input': ['EXA1', 'EXA2', 'EXA3'], 
                'label': ['EXB1', 'EXB2', 'EXB3']
            }
        )
    )
    print()

    print('OUTPUT-ALIGN')
    print(
        icl_prompter.generate_prompt(
            input_exemplar={'input': 'Q1', 'label': 'XXX'},
            output_alignment_prompt='OUTPUT-ALIGNMENT'
        )
    )
    print()

    print('ICL + INPUT-ALIGN')
    print(
        icl_prompter.generate_prompt(
            input_exemplar={'input': 'Q1', 'label': 'XXX'},
            icl_exemplars={
                'input': ['EXA1', 'EXA2', 'EXA3'], 
                'label': ['EXB1', 'EXB2', 'EXB3']
            },
            input_alignment_exemplars={
                'input': ['EXA1', 'EXA2', 'EXA3'], 
                'label': ['EXB1', 'EXB2', 'EXB3']
            }
        )
    )
    print()
    
    print('ICL + OUTPUT-ALIGN')
    print(
        icl_prompter.generate_prompt(
            input_exemplar={'input': 'Q1', 'label': 'XXX'},
            icl_exemplars={
                'input': ['EXA1', 'EXA2', 'EXA3'], 
                'label': ['EXB1', 'EXB2', 'EXB3']
            },
            output_alignment_prompt='OUTPUT-ALIGNMENT'
        )
    )
    print()

    print('ALL')
    print(
        icl_prompter.generate_prompt(
            input_exemplar={'input': 'Q1', 'label': 'XXX'},
            icl_exemplars={
                'input': ['EXA1', 'EXA2', 'EXA3'], 
                'label': ['EXB1', 'EXB2', 'EXB3']
            },
            input_alignment_exemplars={
                'input': ['EXA1', 'EXA2', 'EXA3'], 
                'label': ['EXB1', 'EXB2', 'EXB3']
            },
            output_alignment_prompt='OUTPUT-ALIGNMENT'
        )
    )
    print()
    
    print('== TEST ITC ==')
    itc_prompter = ITCPrompter(headers=['TGT-SENT','LABEL','SRC-SENT'], attribute_keys=['text_1', 'label', 'text_2'])
    
    print('ZERO-SHOT')
    print(
        itc_prompter.generate_prompt(
            input_query='INPUT-QUERY'
        )
    )
    print()
    
    print('FEW-SHOT')
    print(
        itc_prompter.generate_prompt(
            input_query='INPUT-QUERY',
            exemplars={
                'text_1': ['EXA1', 'EXA2', 'EXA3'], 
                'label':['LBL1', 'LBL2', 'LBL3'], 
                'text_2': ['EXB1', 'EXB2', 'EXB3']
            }
        )
    )
    print()