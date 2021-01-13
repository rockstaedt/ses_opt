def print_caption(name:str):
    """
    This function prints a heading to the terminal.

    Args:
        name (str): heading content
    """
    print('###################################################################')
    print(name)
    print('###################################################################')
    print()

def print_title(name:str):
    """
    This function prints a title to the terminal

    Args:
        name (str): subheading content
    """
    print('*******************************************************************')
    print('*******************************************************************')
    print(name)
    print('*******************************************************************')
    print('*******************************************************************')
    print()

def print_convergence(converged:bool):
    """
    This function prints the evaluation of the 'converged' variable to the
    terminal.

    Args:
        converged (bool): [description]
    """
    if not converged:
        print()
        print('--> Not converging. Next iteration.')
        print()
    else:
        print()
        print('--> Converged. Stop algorithm.')
        print()

def print_status(i:int):
    """
    This function prints a dot to the terminal to illustrate the ongoing
    iteration.

    Args:
        i (int): counter of iteration
    """
    # increase by one because index of samples starts with zero
    i += 1
    if i % 100 == 0:
        print(i)
    else:
        print('.', end='')