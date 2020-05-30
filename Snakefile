import sample_le
import numpy as np
proportions = np.linspace(0.0, 0.9, 10)

ortho_kws = {'tstep':0.5,
'nsteps':20,
'nconfig':10000 } 

rule all:
    input: expand("excited.{prop}.chk",prop=proportions)

rule MF_CASCI:
    input: "geometry.xyz"
    output: hf="hf.chk",casci="casci.chk"
    run:
        with open(input[0],'r') as f:
            txt = f.read()
        sample_le.hf_casci(txt,output.hf, output.casci)


rule LINEAR: 
    input: hf="hf.chk",casci="casci.chk"
    output: 'linear.chk'
    run: 
        mc_calc = sample_le.gen_wf(input.hf,input.casci, {0:1.0})
        sample_le.linear(mc_calc, nconfig = 10000, hdf_file=output[0], verbose=True)

rule EXCITED:
    input: hf="hf.chk",casci="casci.chk",anchor='linear.chk'
    output: "excited.{proportion}.chk"
    run:
        mc_ex = sample_le.gen_wf(input.hf,input.casci,{0:float(wildcards.proportion), 1:1-float(wildcards.proportion)})
        mc_ref = sample_le.restart_wf(input.hf, input.casci, input.anchor)
        sample_le.optimize_excited([mc_ref],mc_ex,Starget=[float(wildcards.proportion)],forcing=[2.0],hdf_file=output[0],**ortho_kws)

rule EXCITED2:
    input: hf="hf.chk",casci="casci.chk",anchor1='linear.chk', anchor2='excited.0.0.chk'
    output: "excited_2.{proportion}.chk"
    run:
        mc_ex = sample_le.gen_wf(input.hf,input.casci,{0:float(wildcards.proportion), 1:0, 2:1-float(wildcards.proportion)})
        mc_ref = sample_le.restart_wf(input.hf, input.casci, input.anchor1)
        mc_ref2 = sample_le.restart_wf(input.hf, input.casci, input.anchor2)
        sample_le.optimize_excited([mc_ref,mc_ref2],mc_ex,Starget=[float(wildcards.proportion),0.0],forcing=[2.0,2.0],hdf_file=output[0],**ortho_kws)


rule EXCITED3:
    input: hf="hf.chk",casci="casci.chk",anchor1='linear.chk', anchor2='excited.0.0.chk', anchor3='excited_2.0.0.chk'
    output: "excited_3.{proportion}.chk"
    run:
        mc_ex = sample_le.gen_wf(input.hf,input.casci,{0:float(wildcards.proportion), 1:0.0, 2:0.0, 3:1-float(wildcards.proportion)})
        mc_ref = sample_le.restart_wf(input.hf, input.casci, input.anchor1)
        mc_ref2 = sample_le.restart_wf(input.hf, input.casci, input.anchor2)
        mc_ref3 = sample_le.restart_wf(input.hf, input.casci, input.anchor2)
        sample_le.optimize_excited([mc_ref,mc_ref2,mc_ref3],mc_ex,Starget=[float(wildcards.proportion),0.0,0.0],forcing=[2.0,2.0,2.0],hdf_file=output[0], **ortho_kws)
