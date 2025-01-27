import termcolor
import tabulate

tabulate.PRESERVE_WHITESPACE = True

def format_table(rows, header=None, footer=None, filter=(), color='green', 
                 min_widths=[], max_widths=[30,55], attrs=None):
    """
    Print a table from a list[list] of rows/columns, or a 2-column dict 
    where the keys are column 1, and the values are column 2.
    
    Header is a list of columns or rows that are inserted at the top.
    Footer is a list of columns or rows that are added to the end.
    
    color names and style attributes are from termcolor library:
      https://github.com/termcolor/termcolor#text-properties
    """
    if isinstance(rows, dict):
        rows = flatten_rows(rows, filter=filter)

    for row in rows:
        for c, col in enumerate(row):
            col = str(col)
            if c < len(min_widths) and len(col) < min_widths[c]:
                col = col + ' ' * (min_widths[c] - len(col))
            if c < len(max_widths) and len(col) > max_widths[c]:
                col = col[:max_widths[c]]
            row[c] = col

    if header:
        if not isinstance(header[0], list):
            header = [header]
        rows = header + rows
        
    if footer:
        if not isinstance(footer[0], list):
            footer = [footer]
        rows = rows + footer
        
    table = tabulate.tabulate(rows, tablefmt='rounded_outline', numalign='center')

    if not color:
        return table

    return '\n'.join([
        termcolor.colored(x, color, attrs=attrs)
        for x in table.split('\n')
    ])

def flatten_rows(seq, filter=()):
    """
    Recursively convert a tree of list/dict/tuple objects to a flat list.
    This is so they can be printed in a simple two-column table.
    """
    def flatten(seq, indent='', prefix='', out=[]):
        iter = range(len(seq)) if isinstance(seq,(list,tuple)) else seq
        for key in iter:
            val = filter(seq,key) if filter else seq[key]
            if not val:
                continue
            if isinstance(seq,dict) and isinstance(val,list):
                flatten(val, indent, f'{key} ', out)
            elif isinstance(val, (tuple,list,dict,map)):
                out.append([indent + prefix + str(key), ''])
                flatten(val, indent + (' ├ ' if len(val) > 1 else ''), out=out)  # ┣
            else:
                out.append([indent + prefix + str(key), val])
        return out
    return flatten(seq)      
