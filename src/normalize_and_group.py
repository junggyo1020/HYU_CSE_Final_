# taassc의 결과 파일을 평균화하는 모듈.
# 이 모듈은 지표들을 평균을 내어 그룹화 하고
# 결과를 기존 CSV 파일에 덮어씀.

import os
import pandas as pd
import numpy as np

# 하이퍼볼릭 탄젠트 기반 정규화 함수 정의
def normalize_with_tanh(value, scale=1):
    if value == 0:
        return 0
    else:
        return np.tanh(value * scale)

# 카테고리별 지표 그룹 정의
CATEGORIES = {
    "Syntactic Components": [
        "nsubj_per_cl", "nsubjpass_per_cl", "dobj_per_cl", "iobj_per_cl",
        "acomp_per_cl", "csubj_per_cl", "csubjpass_per_cl", "ncomp_per_cl"
    ],
    "Clause Structure": [
        "advcl_per_cl", "ccomp_per_cl", "xcomp_per_cl", "xsubj_per_cl",
        "conj_per_cl", "cc_per_cl", "parataxis_per_cl"
    ],
    "Dependency and Relations": [
        "cl_av_deps", "cl_ndeps_std_dev", "dep_per_cl", "agent_per_cl", "expl_per_cl"
    ],
    "Modifiers": [
        "advmod_per_cl", "prt_per_cl", "prep_per_cl", "prepc_per_cl",
        "neg_per_cl", "mark_per_cl", "tmod_per_cl"
    ],
    "Auxiliaries and Modals": [
        "aux_per_cl", "auxpass_per_cl", "modal_per_cl"
    ],
    "Discourse Markers": [
        "discourse_per_cl"
    ]
}

# 디렉토리 내 모든 CSV 파일을 처리하고 새로운 파일로 대체하는 함수
def normalize_and_group(input_directory):
    for filename in os.listdir(input_directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_directory, filename)
            df = pd.read_csv(file_path)

            # filename 열이 있으면 보존
            if 'filename' not in df.columns:
                df['filename'] = filename

            # 모든 지표에 대해 정규화 수행
            for column in df.columns:
                if column in sum(CATEGORIES.values(), []):  # 모든 지표 리스트에 포함된 열만 정규화
                    df[column] = df[column].apply(lambda x: normalize_with_tanh(x))

            # 카테고리별 평균 계산
            category_averages = {}
            for category, indicators in CATEGORIES.items():
                # 각 카테고리의 평균 계산 후 저장
                category_averages[category] = df[indicators].mean(axis=1)

            # filename과 함께 카테고리별 평균 저장
            result_df = pd.DataFrame(category_averages)
            result_df.insert(0, 'filename', df['filename'])  # filename 열을 가장 왼쪽에 추가

            # 기존 파일을 정규화 및 평균화된 결과로 대체
            result_df.to_csv(file_path, index=False)
            print(f"처리 완료: {file_path}")

# 예시 사용법
normalize_and_group("csv_files/taassc_result")
