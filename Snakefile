import sample_le
import numpy as np
samples = range(0,10)

ortho_kws = {'tstep':0.5,
'nsteps':20,
'nconfig':500 } 

linear_kws = {'nconfig':500 } 

rule all:
    input: "vmc_tbdm_excited_3.0.0.chk",expand("vmc_tbdm_sample_{number}.chk",number=samples)

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
        sample_le.linear(mc_calc, hdf_file=output[0], verbose=True, **linear_kws)

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
        mc_ref3 = sample_le.restart_wf(input.hf, input.casci, input.anchor3)
        sample_le.optimize_excited([mc_ref,mc_ref2,mc_ref3],mc_ex,Starget=[float(wildcards.proportion),0.0,0.0],forcing=[2.0,2.0,2.0],hdf_file=output[0], **ortho_kws)


rule RANDOM_SUPERPOSITION:
    input: hf="hf.chk",casci="casci.chk",anchor1='linear.chk', anchor2='excited.0.0.chk', anchor3='excited_2.0.0.chk'
    output: "sample_{samplenumber}.chk"
    run:
        props = np.random.rand(4)
        props = props/np.linalg.norm(props)
        mc_ex = sample_le.gen_wf(input.hf,input.casci,{0:props[0], 1:props[1], 2:props[2], 3:props[3]})
        mc_ref = sample_le.restart_wf(input.hf, input.casci, input.anchor1)
        mc_ref2 = sample_le.restart_wf(input.hf, input.casci, input.anchor2)
        mc_ref3 = sample_le.restart_wf(input.hf, input.casci, input.anchor3)
        sample_le.optimize_excited([mc_ref,mc_ref2,mc_ref3],mc_ex,Starget=props[0:3],forcing=[2.0,2.0,2.0],hdf_file=output[0], **ortho_kws)

rule EVAL_BIG1RDM:
    input: hf="hf.chk",casci="casci.chk",wf="{fname}.chk"
    output: "big1rdm_{fname}.chk"
    run: 
        mc_eval = sample_le.restart_wf(input.hf, input.casci, input.wf)
        sample_le.evaluate_big1rdm(mc_eval, output[0], nsteps=500)


rule CONVERT_BASIS:
    input: hf="hf.chk", casci="casci.chk", hdf_vmc = "big1rdm_{fname}.chk"
    output: "smallbasis_{fname}.chk"
    run:
        # Don't need the S-J wave function here, just reading in the mean field and mol objects
        mc_tmp = sample_le.gen_wf(input.hf, input.casci, {0:1, 1:0, 2:0, 3:0})
        sample_le.convert_basis(mc_tmp['mol'],mc_tmp['mf'],input.hdf_vmc,output[0])


rule EVAL_TBDM:
    input: hf="hf.chk", casci="casci.chk", wf="{fname}.chk", smallbasis = "smallbasis_{fname}.chk"
    output: "vmc_tbdm_{fname}.chk"
    run:
        # Don't need the S-J wave function here, just reading in the mean field and mol objects
        mc_tmp = sample_le.restart_wf(input.hf, input.casci, input.wf)
        sample_le.evaluate_smallbasis(mc_tmp, input.smallbasis, hdf_file=output[0])