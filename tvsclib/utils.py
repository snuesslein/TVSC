import numpy as np
import matplotlib.pyplot as plt
import tvsclib



def show_system(system):
    """show_system display a graphical representation of the system

    function that uses matshow to display the resulting matrix
    and also shows the divisions of the input and output

        Args:
            sysstem [StrictSystem or MixedSystem]: System to display

    TODO: possible additioanl fature: mark D with different color/linstyle
    """
    mat = system.to_matrix()
    plt.matshow(mat)
    x=-0.5
    y=-0.5
    if type(system)==tvsclib.mixed_system.MixedSystem:
        for d_in in system.dims_in:
            x+=d_in
            plt.vlines(x,-0.5,mat.shape[0]-0.5)

        for d_out in system.dims_out:
            y+=d_out
            plt.hlines(y,-0.5,mat.shape[0]-0.5)

    elif system.causal:
        for st in system.stages:
            x+=st.dim_in
            plt.hlines(y,-0.5,x)
            plt.vlines(x,y,mat.shape[0]-0.5)
            y+=st.dim_out
    else:
        for st in system.stages:
            y+=st.dim_out
            plt.hlines(y,x,mat.shape[1]-0.5)
            plt.vlines(x,-0.5,y)
            x+=st.dim_in


def check_dims(system,dim_state_in=0,dim_state_out=0,text_output=True,return_report=False):
    """check_dims Test if the dimentions of the matrices are correct
        Args:
            system (StrictSystem): Causal system to check
            dim_state_in (int): input state of first dim (default is 0)
            dim_state_out (int): output state of last dim (default is 0)
            text_output (bool): if True the function prints the result
            return_report (bool): if True the function returns a report as string

        Returns:
            Bool: True if matrix shapes are correct Fasle otherwise
            Str: Report, is only returend if return_report=True is set

    """
    rep = ""
    correct = True
    dim_state = dim_state_in
    #iterate up or down depending on causal/anticausal, the rest stays the same
    if system.causal:
        it = range(len(system.stages))
    else:
        it = range(len(system.stages)-1,-1,-1)
    for i in it:
        st = system.stages[i]
        #check if the state input is correct for A and C
        if st.A_matrix.shape[1] != dim_state:
            correct = False
            rep = rep + "Problem at index "+str(i)+": State dims of A do not match: old:"+str(dim_state)+ \
                  " new: "+str(st.A_matrix.shape[1])+"\n"
        if st.C_matrix.shape[1] != dim_state:
            correct = False
            rep = rep + "Problem at index "+str(i)+": State dims of C do not match: old:"+str(dim_state)+ \
                  " new: "+str(st.C_matrix.shape[1])+"\n"

        #check if the state output of A and B match
        dim_state = st.A_matrix.shape[0]
        if st.B_matrix.shape[0] != dim_state:
            correct = False
            rep = rep + "Problem at index "+str(i)+": State dims of A and B do not match: A:"+str(dim_state)+ \
                  "B: "+str(st.B_matrix.shape[0]) + "\n"

        #check if the input dims match
        if st.B_matrix.shape[1] != st.D_matrix.shape[1]:
            correct = False
            rep = rep + "Problem at index "+str(i)+": Input dims of B and D do not match: B:"+str(st.B_matrix.shape[1])+ \
                  "D: "+str(st.D_matrix.shape[1]) +"\n"

        #check if the output states match
        if st.C_matrix.shape[0] != st.D_matrix.shape[0]:
            correct = False
            rep = rep + "Problem at index "+str(i)+": Output dims of C and D do not match: C:"+str(st.C_matrix.shape[0])+ \
                  "D: "+str(st.D_matrix.shape[0]) +"\n"
    if dim_state != dim_state_out:
        correct = False
        rep = rep + "final state dim does not match"
    if text_output:
        if correct:
            print("Matrix shapes are correct")
        else:
            print("Matrix shapes are not correct")
            print(rep)
    if return_report:
        return correct,rep
    else:
        return correct
