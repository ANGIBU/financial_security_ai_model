import pickle
import json
import os

# 변환할 pkl 파일 목록
pkl_files = [
    r"C:\Users\facec\바탕화면\AI\financial_security_ai_model\pkl\processing_history.pkl",
    r"C:\Users\facec\바탕화면\AI\financial_security_ai_model\pkl\learning_data.pkl",
    r"C:\Users\facec\바탕화면\AI\financial_security_ai_model\pkl\analysis_history.pkl"
]

output_dir = r"C:\Users\facec\바탕화면\AI\financial_security_ai_model\pkl\json_output"
os.makedirs(output_dir, exist_ok=True)

for pkl_path in pkl_files:
    try:
        # PKL 불러오기
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        # 파일 이름 추출
        filename = os.path.splitext(os.path.basename(pkl_path))[0]

        # 콘솔에 데이터 출력
        print(f"\n=== {filename}.pkl 내용 ===")
        print(data)

        # JSON 저장
        json_path = os.path.join(output_dir, f"{filename}.json")
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(data, jf, ensure_ascii=False, indent=4)

        print(f"→ {json_path} 로 저장 완료")

    except Exception as e:
        print(f"[ERROR] {pkl_path} 읽기 실패: {e}")
