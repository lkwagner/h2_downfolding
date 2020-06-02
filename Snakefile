import sample_le
import numpy as np

ortho_kws = {'tstep':0.5,
'nsteps':20,
'nconfig':500 } 

linear_kws = {'nconfig':500 } 
radii = [1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.2,3.4,3.6,3.8,4.0]
rule all:
    input: expand("r{r}/vmc_tbdm_eigenstate{n}.chk",n=[0,1,2,3],r=radii),
           expand("r{r}/vmc_tbdm_sample_{number}.chk",number=range(0,30),r=radii)

rule GEOMETRY:
    output: "r{bond}/geometry.xyz"
    run:
        with open(output[0],'w') as f:
            f.write(f"H 0. 0. 0.; H 0. 0. {wildcards.bond}")


rule MF_CASCI:
    input: "r{bond}/geometry.xyz"
    output: hf="r{bond}/hf.chk",casci="r{bond}/casci.chk"
    run:
        with open(input[0],'r') as f:
            txt = f.read()
        sample_le.hf_casci(txt,output.hf, output.casci)


rule LINEAR: 
    input: hf="r{bond}/hf.chk",casci="r{bond}/casci.chk"
    output: 'r{bond}/eigenstate0.chk'
    run: 
        mc_calc = sample_le.gen_wf(input.hf,input.casci, {0:1.0})
        sample_le.linear(mc_calc, hdf_file=output[0], verbose=True, **linear_kws)

rule EXCITED1:
    input: hf="r{bond}/hf.chk",casci="r{bond}/casci.chk",anchor="r{bond}/eigenstate0.chk"
    output: "r{bond}/eigenstate1.chk"
    run:
        mc_ex = sample_le.gen_wf(input.hf,input.casci,{0:0.0,1:1.0})
        mc_ref = sample_le.restart_wf(input.hf, input.casci, input.anchor)
        sample_le.optimize_excited([mc_ref],mc_ex,Starget=[0.0],forcing=[2.0],hdf_file=output[0],**ortho_kws)

rule EXCITED2:
    input: hf="r{bond}/hf.chk",casci="r{bond}/casci.chk",anchor1="r{bond}/eigenstate0.chk",anchor2="r{bond}/eigenstate1.chk"
    output: "r{bond}/eigenstate2.chk"
    run:
        mc_ex = sample_le.gen_wf(input.hf,input.casci,{0:0., 1:0, 2:1.})
        mc_ref = sample_le.restart_wf(input.hf, input.casci, input.anchor1)
        mc_ref2 = sample_le.restart_wf(input.hf, input.casci, input.anchor2)
        sample_le.optimize_excited([mc_ref,mc_ref2],mc_ex,Starget=[0.0,0.0],forcing=[2.0,2.0],hdf_file=output[0],**ortho_kws)


rule EXCITED3:
    input: hf="r{bond}/hf.chk",casci="r{bond}/casci.chk",anchor1="r{bond}/eigenstate0.chk", anchor2="r{bond}/eigenstate1.chk", anchor3="r{bond}/eigenstate2.chk"
    output: "r{bond}/eigenstate3.chk"
    run:
        mc_ex = sample_le.gen_wf(input.hf,input.casci,{0:0, 1:0.0, 2:0.0, 3:1.0})
        mc_ref = sample_le.restart_wf(input.hf, input.casci, input.anchor1)
        mc_ref2 = sample_le.restart_wf(input.hf, input.casci, input.anchor2)
        mc_ref3 = sample_le.restart_wf(input.hf, input.casci, input.anchor3)
        sample_le.optimize_excited([mc_ref,mc_ref2,mc_ref3],mc_ex,Starget=[0.0,0.0,0.0],forcing=[2.0,2.0,2.0],hdf_file=output[0], **ortho_kws)


rule RANDOM_SUPERPOSITION:
    input: hf="r{bond}/hf.chk",casci="r{bond}/casci.chk",anchor1="r{bond}/eigenstate0.chk", anchor2="r{bond}/eigenstate1.chk", anchor3="r{bond}/eigenstate2.chk"
    output: "r{bond}/sample_{samplenumber}.chk"
    run:
        props = np.random.rand(4)
        props = props/np.linalg.norm(props)
        mc_ex = sample_le.gen_wf(input.hf,input.casci,{0:props[0], 1:props[1], 2:props[2], 3:props[3]})
        mc_ref = sample_le.restart_wf(input.hf, input.casci, input.anchor1)
        mc_ref2 = sample_le.restart_wf(input.hf, input.casci, input.anchor2)
        mc_ref3 = sample_le.restart_wf(input.hf, input.casci, input.anchor3)
        sample_le.optimize_excited([mc_ref,mc_ref2,mc_ref3],mc_ex,Starget=props[0:3],forcing=[2.0,2.0,2.0],hdf_file=output[0], **ortho_kws)

rule EVAL_BIG1RDM:
    input: hf="r{bond}/hf.chk",casci="r{bond}/casci.chk",wf="r{bond}/{fname}.chk"
    output: "r{bond}/big1rdm_{fname}.chk"
    run: 
        mc_eval = sample_le.restart_wf(input.hf, input.casci, input.wf)
        sample_le.evaluate_big1rdm(mc_eval, output[0], nsteps=500)


rule CONVERT_BASIS:
    input: hf="r{bond}/hf.chk", casci="r{bond}/casci.chk", hdf_vmc = "r{bond}/big1rdm_{fname}.chk"
    output: "r{bond}/smallbasis_{fname}.chk"
    run:
        # Don't need the S-J wave function here, just reading in the mean field and mol objects
        mc_tmp = sample_le.gen_wf(input.hf, input.casci, {0:1, 1:0, 2:0, 3:0})
        sample_le.convert_basis(mc_tmp['mol'],mc_tmp['mf'],input.hdf_vmc,output[0])


rule EVAL_TBDM:
    input: hf="r{bond}/hf.chk", casci="r{bond}/casci.chk", wf="r{bond}/{fname}.chk", smallbasis = "r{bond}/smallbasis_{fname}.chk"
    output: "r{bond}/vmc_tbdm_{fname}.chk"
    run:
        # Don't need the S-J wave function here, just reading in the mean field and mol objects
        mc_tmp = sample_le.restart_wf(input.hf, input.casci, input.wf)
        sample_le.evaluate_smallbasis(mc_tmp, input.smallbasis, hdf_file=output[0], nsteps=1000)