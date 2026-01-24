1) AR vs BI
keys: A=1905, B=1905, common=1905, onlyA/B=0
shape/dtype mismatch: 0/0
global: mean_abs=7.9507e-04, rmse=0.001213, rel_rmse_vs_a=0.032361, max_abs=0.053401
2) AR vs HunyuanVideo-1.5 ref
keys: AR=1905, ref=1793, common=1793, onlyAR=112, onlyRef=0
shape/dtype mismatch: 0/0
global: mean_abs=9.5281e-04, rmse=0.001471, rel_rmse_vs_a=0.038724, max_abs=0.136999
only in AR 的典型项（说明 HY-WorldPlay 相对 ref 多了 video/action 相关或额外模块）：action_in.*、大量 double_blocks.*.img_attn_prope_proj.* 等
3) BI vs HunyuanVideo-1.5 ref
keys: BI=1905, ref=1793, common=1793, onlyBI=112, onlyRef=0（同 AR）
shape/dtype mismatch: 0/0
global: mean_abs=7.4390e-04, rmse=0.001138, rel_rmse_vs_a=0.029981, max_abs=0.105675


                    Base model 



                             BI 
                       
                       AR