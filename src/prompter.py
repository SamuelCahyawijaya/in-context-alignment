###
# In-Context Learning with Alignment
###
class ICLPrompter(object):
    __slots__ = ("instruction_template", "separator")

    def __init__(self, instruction_template: str = "", icl_key: str | list[str], iia_key: str | list[str], separator: str = "\n") -> None:
        self.instruction_template = instruction_template
        self.icl_key = icl_key
        self.iia_key = iia_key
        self.separator = separator

        assert '[CONTEXT]' in self.instruction_template
        assert '[QUERY]' in self.instruction_template

    def generate_prompt(self, 
        input_exemplar: None | dict = None, 
        icl_exemplars: None | dict = None,
        icl_template: None | str = None,
        input_alignment_exemplars: None | dict = None,
        input_alignment_template: None | str = None,
        output_alignment_prompt: None | str = None
    ) -> str:
        # Init Prompt & Contexts
        prompt = self.instruction_template
        contexts = []

        # Format ICL
        if icl_exemplars is not None:
            for ex in zip(*icl_exemplars):
                feats = [ex[key] for key in index_key + ['label']] 
                contexts.append(icl_template.format(*feats))

        # Format Input Alignment ICL
        if input_alignment_exemplars is not None:
            for x, y in zip(*input_alignment_exemplars):
                contexts.append(input_alignment_template.format(x, y))                        

        # Format Label Alignment ICL
        if output_alignment_prompt is not None:
            contexts.append(output_alignment_prompt)

        if len(contexts) > 0:
            context = '\n'.join(contexts)
            prompt = prompt.replace('[CONTEXT]', context)
        else:
            # Remove `[CONTEXT]`
            prompt = prompt.replace('[CONTEXT]', '')

        # Format Input Query
        icl_template.format()
        prompt = prompt.replace('[QUERY]', input_query)

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
        rows.append(f'| {input_query} |')

        # Return Resulting Prompt
        prompt = self.header_template + '\n'.join(rows)
        return prompt
    
if __name__ == '__main__':
    icl_prompter = ICLPrompter(instruction_template="What is the sentiment of the following sentences?\n[CONTEXT]\n[INPUT] => [LABELS_CHOICE]")
    
    print('== TEST ICL ==')
    print('ZERO-SHOT')
    print(
        icl_prompter.generate_prompt(input_query='INPUT-QUERY')
    )
    print()
    
    print('ICL')
    print(
        icl_prompter.generate_prompt(
            input_query='INPUT-QUERY',
            icl_exemplars=(['EX1', 'EX2', 'EX3'], ['LBL1', 'LBL2', 'LBL3']),
            icl_template='{} => {}',
        )
    )    
    print()

    print('INPUT-ALIGN')
    print(
        icl_prompter.generate_prompt(
            input_query='INPUT-QUERY',
            input_alignment_exemplars=(['EXA1', 'EXA2', 'EXA3'], ['EXB1', 'EXB2', 'EXB3']),
            input_alignment_template='{} => {}',
        )
    )
    print()

    print('OUTPUT-ALIGN')
    print(
        icl_prompter.generate_prompt(
            input_query='INPUT-QUERY',
            output_alignment_prompt='OUTPUT-ALIGNMENT'
        )
    )
    print()

    print('ICL + INPUT-ALIGN')
    print(
        icl_prompter.generate_prompt(
            input_query='INPUT-QUERY',
            icl_exemplars=(['EX1', 'EX2', 'EX3'], ['LBL1', 'LBL2', 'LBL3']),
            icl_template='{} => {}',
            input_alignment_exemplars=(['EXA1', 'EXA2', 'EXA3'], ['EXB1', 'EXB2', 'EXB3']),
            input_alignment_template='{} => {}',
        )
    )
    print()
    
    print('ICL + OUTPUT-ALIGN')
    print(
        icl_prompter.generate_prompt(
            input_query='INPUT-QUERY',
            icl_exemplars=(['EX1', 'EX2', 'EX3'], ['LBL1', 'LBL2', 'LBL3']),
            icl_template='{} => {}',
            output_alignment_prompt='OUTPUT-ALIGNMENT'
        )
    )
    print()

    print('ALL')
    print(
        icl_prompter.generate_prompt(
            input_query='INPUT-QUERY',
            icl_exemplars=(['EX1', 'EX2', 'EX3'], ['LBL1', 'LBL2', 'LBL3']),
            icl_template='{} => {}',
            input_alignment_exemplars=(['EXA1', 'EXA2', 'EXA3'], ['EXB1', 'EXB2', 'EXB3']),
            input_alignment_template='{} => {}',
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