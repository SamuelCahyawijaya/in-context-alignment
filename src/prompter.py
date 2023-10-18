###
# In-Context Learning with Alignment
###
class ICLPrompter(object):
    __slots__ = ("instruction_template", "separator")

    def __init__(self, template: str = "", separator: str = "\n") -> None:
        self.instruction_template = instruction_template
        self.separator = separator

        assert '[CONTEXT]' in self.instruction_template
        assert '[INPUT]' in self.instruction_template
        assert '[LABELS_CHOICE]' in self.instruction_template

    def generate_prompt(self, 
        input_query: None | str = None, 
        icl_exemplars: None | tuple(list, list) = None,
        icl_template: None | str = None,
        input_alignment_exemplars: None | tuple(list, list) = None,
        input_alignment_template: None | str = None,
        output_alignment_prompt: None | str = None
    ) -> str:
        # Init Prompt & Contexts
        prompt = self.instruction_template
        contexts = []

        # Format ICL
        if icl_exemplars is not None:
            for x, y in zip(*icl_exemplars):
                if icl_template is None:
                    contexts.append(f'{x} => {y}')
                else:
                    contexts.append(icl_template.format(x, y))

        # Format Input Alignment ICL
        if input_alignment_exemplars is not None:
            for x, y in zip(*input_alignment_exemplars:)
                if input_alignment_template is None:
                    contexts.append(f'{x} => {y}')
                else:
                    contexts.append(input_alignment_template.format(x, y))                        

        # Format Label Alignment ICL
        if output_alignment_prompt is not None:
            contexts.append(output_alignment_prompt)

        if len(contexts) > 0:
            context = '\n'.join(contexts)
            prompt = prompt.replace('[CONTEXT]', context)
        else:
            # Remove `[ICL_EXEMPLARS]`
            prompt = prompt.replace('[CONTEXT]', '')

        # Format Input Query
        prompt = prompt.replace('[INPUT]', input_query)

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
            header_separator += f"|{'|'.join('---' for _ in self.headers)}|\n"
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
                row_dict = {key: exemplars[key][i]} for key in self.attribute_keys:
                rows.append(self.row_template.format(**row_dict))

        # Format Input Query
        rows.append(f'| {input_query} |')

        # Return Resulting Prompt
        prompt = self.header_template + self.row_template + '\n'.join(rows)
        return prompt
