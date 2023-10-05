import torch
import torch.nn as nn
from network import modelA, modelB
import thop

if __name__ == "__main__":
    # modelA
    model_a = modelA()

    # modelB
    model_b = modelB()

    input = torch.randn(16, 3, 28, 28)

    # 연산량 및 파라미터 수 계산
    modelA_flops, modelA_params = thop.profile(model_a, inputs=(input,))
    modelB_flops, modelB_params = thop.profile(model_b, inputs=(input,))

    print("modelA 연산량 (GFLOPs):", modelA_flops)
    print("modelA 파라미터 수 (Millions):", modelA_params)
    print("modelB 연산량 (GFLOPs):", modelB_flops)
    print("modelB 파라미터 수 (Millions):", modelB_params)