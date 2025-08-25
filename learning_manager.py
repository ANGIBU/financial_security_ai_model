# learning_manager.py

import os
import pickle
import json
import time
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import pandas as pd

from config import PKL_DIR, MEMORY_CONFIG


class LearningManager:

    def __init__(self):
        self.pkl_dir = PKL_DIR
        self.pkl_dir.mkdir(exist_ok=True)
        
        self.learning_data = {
            "question_analysis": {},
            "answer_patterns": {},
            "domain_accuracy": {},
            "mc_patterns": {},
            "successful_answers": [],
            "failed_answers": [],
            "quality_scores": {},
        }
        
        self.load_existing_data()

    def load_existing_data(self):
        """기존 학습 데이터 로드"""
        try:
            learning_file = self.pkl_dir / "learning_data.pkl"
            if learning_file.exists():
                with open(learning_file, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.learning_data.update(saved_data)
                    print(f"기존 학습 데이터 로드: {len(self.learning_data['successful_answers'])}개 성공 답변")
        except Exception as e:
            print(f"학습 데이터 로드 실패: {e}")

    def save_learning_data(self):
        """학습 데이터 저장"""
        try:
            learning_file = self.pkl_dir / "learning_data.pkl"
            with open(learning_file, 'wb') as f:
                pickle.dump(self.learning_data, f)
            
            # JSON 백업도 저장
            backup_file = self.pkl_dir / f"learning_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(self._prepare_json_data(), f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"학습 데이터 저장 실패: {e}")

    def _prepare_json_data(self) -> Dict:
        """JSON 저장을 위한 데이터 변환"""
        json_data = {}
        for key, value in self.learning_data.items():
            if isinstance(value, dict):
                json_data[key] = value
            elif isinstance(value, list):
                json_data[key] = value[:100]  # 최근 100개만 저장
            else:
                json_data[key] = str(value)
        return json_data

    def record_question_analysis(self, question_id: str, question: str, 
                               question_type: str, domain: str, intent_analysis: Dict):
        """질문 분석 결과 기록"""
        self.learning_data["question_analysis"][question_id] = {
            "question": question,
            "type": question_type,
            "domain": domain,
            "intent": intent_analysis,
            "timestamp": time.time()
        }

    def record_answer_attempt(self, question_id: str, question: str, answer: str, 
                            question_type: str, domain: str, is_successful: bool, 
                            method_used: str = "llm"):
        """답변 시도 결과 기록"""
        attempt_data = {
            "question_id": question_id,
            "question": question,
            "answer": answer,
            "type": question_type,
            "domain": domain,
            "method": method_used,
            "timestamp": time.time(),
            "length": len(answer) if answer else 0
        }

        if is_successful:
            self.learning_data["successful_answers"].append(attempt_data)
            # 최대 개수 제한
            max_successful = MEMORY_CONFIG["max_learning_records"]["successful_answers"]
            if len(self.learning_data["successful_answers"]) > max_successful:
                self.learning_data["successful_answers"] = self.learning_data["successful_answers"][-max_successful:]
        else:
            self.learning_data["failed_answers"].append(attempt_data)
            # 최대 개수 제한
            max_failed = MEMORY_CONFIG["max_learning_records"]["failed_answers"]
            if len(self.learning_data["failed_answers"]) > max_failed:
                self.learning_data["failed_answers"] = self.learning_data["failed_answers"][-max_failed:]

        # 도메인별 정확도 업데이트
        if domain not in self.learning_data["domain_accuracy"]:
            self.learning_data["domain_accuracy"][domain] = {"success": 0, "total": 0}
        
        self.learning_data["domain_accuracy"][domain]["total"] += 1
        if is_successful:
            self.learning_data["domain_accuracy"][domain]["success"] += 1

    def record_mc_pattern(self, question: str, correct_answer: str, pattern_type: str = ""):
        """객관식 패턴 학습"""
        pattern_key = self._extract_mc_pattern_key(question)
        
        if pattern_key not in self.learning_data["mc_patterns"]:
            self.learning_data["mc_patterns"][pattern_key] = {
                "answers": [],
                "pattern_type": pattern_type,
                "count": 0
            }
        
        self.learning_data["mc_patterns"][pattern_key]["answers"].append(correct_answer)
        self.learning_data["mc_patterns"][pattern_key]["count"] += 1

    def _extract_mc_pattern_key(self, question: str) -> str:
        """객관식 패턴 키 추출"""
        question_lower = question.lower()
        
        if "해당하지 않는" in question_lower:
            if "금융투자업" in question_lower:
                return "금융투자_해당하지않는"
            elif "위험관리" in question_lower:
                return "위험관리_해당하지않는"
            else:
                return "일반_해당하지않는"
        elif "적절하지 않은" in question_lower:
            return "일반_적절하지않은"
        elif "가장 적절한" in question_lower:
            return "일반_가장적절한"
        elif "중요한 요소" in question_lower:
            return "일반_중요한요소"
        else:
            return "일반_객관식"

    def get_domain_patterns(self, domain: str) -> List[Dict]:
        """도메인별 성공 패턴 조회"""
        domain_patterns = []
        
        for answer_data in self.learning_data["successful_answers"]:
            if answer_data["domain"] == domain:
                domain_patterns.append({
                    "question": answer_data["question"],
                    "answer": answer_data["answer"],
                    "method": answer_data["method"]
                })
        
        return domain_patterns[-10:]  # 최근 10개

    def get_mc_prediction(self, question: str, max_choice: int) -> str:
        """객관식 답변 예측"""
        pattern_key = self._extract_mc_pattern_key(question)
        
        if pattern_key in self.learning_data["mc_patterns"]:
            answers = self.learning_data["mc_patterns"][pattern_key]["answers"]
            if answers:
                # 가장 자주 나온 답변 반환
                answer_counts = {}
                for ans in answers:
                    if ans.isdigit() and 1 <= int(ans) <= max_choice:
                        answer_counts[ans] = answer_counts.get(ans, 0) + 1
                
                if answer_counts:
                    best_answer = max(answer_counts.items(), key=lambda x: x[1])[0]
                    return best_answer
        
        return None

    def get_successful_template(self, domain: str, intent_type: str) -> str:
        """성공한 답변에서 템플릿 추출"""
        matching_answers = []
        
        for answer_data in self.learning_data["successful_answers"]:
            if (answer_data["domain"] == domain and 
                answer_data["type"] == "subjective" and
                len(answer_data["answer"]) > 50):
                matching_answers.append(answer_data["answer"])
        
        if matching_answers:
            # 가장 최근 성공 답변 반환
            return matching_answers[-1]
        
        return None

    def calculate_domain_accuracy(self, domain: str) -> float:
        """도메인별 정확도 계산"""
        if domain in self.learning_data["domain_accuracy"]:
            domain_data = self.learning_data["domain_accuracy"][domain]
            if domain_data["total"] > 0:
                return domain_data["success"] / domain_data["total"]
        return 0.0

    def get_learning_stats(self) -> Dict:
        """학습 통계 정보"""
        total_successful = len(self.learning_data["successful_answers"])
        total_failed = len(self.learning_data["failed_answers"])
        total_attempts = total_successful + total_failed
        
        stats = {
            "total_attempts": total_attempts,
            "successful_attempts": total_successful,
            "failed_attempts": total_failed,
            "success_rate": total_successful / total_attempts if total_attempts > 0 else 0.0,
            "domain_accuracies": {}
        }
        
        for domain, data in self.learning_data["domain_accuracy"].items():
            if data["total"] > 0:
                stats["domain_accuracies"][domain] = data["success"] / data["total"]
        
        return stats

    def cleanup_old_data(self):
        """오래된 학습 데이터 정리"""
        current_time = time.time()
        one_month_ago = current_time - (30 * 24 * 60 * 60)  # 30일 전
        
        # 오래된 질문 분석 데이터 제거
        old_questions = []
        for q_id, data in self.learning_data["question_analysis"].items():
            if data.get("timestamp", current_time) < one_month_ago:
                old_questions.append(q_id)
        
        for q_id in old_questions:
            del self.learning_data["question_analysis"][q_id]
        
        # 오래된 답변 데이터 제거
        def filter_old_answers(answers_list):
            return [ans for ans in answers_list 
                   if ans.get("timestamp", current_time) >= one_month_ago]
        
        self.learning_data["successful_answers"] = filter_old_answers(
            self.learning_data["successful_answers"]
        )
        self.learning_data["failed_answers"] = filter_old_answers(
            self.learning_data["failed_answers"]
        )

    def export_analysis(self, output_file: str = None):
        """학습 분석 결과 내보내기"""
        if not output_file:
            output_file = self.pkl_dir / f"learning_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        stats = self.get_learning_stats()
        
        analysis_text = f"""
학습 데이터 분석 보고서
생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

=== 전체 통계 ===
총 시도 횟수: {stats['total_attempts']}
성공 횟수: {stats['successful_attempts']}
실패 횟수: {stats['failed_attempts']}
전체 성공률: {stats['success_rate']:.3f}

=== 도메인별 정확도 ===
"""
        
        for domain, accuracy in stats["domain_accuracies"].items():
            analysis_text += f"{domain}: {accuracy:.3f}\n"
        
        analysis_text += f"""
=== 객관식 패턴 ===
학습된 패턴 수: {len(self.learning_data['mc_patterns'])}
"""
        
        for pattern, data in self.learning_data["mc_patterns"].items():
            analysis_text += f"{pattern}: {data['count']}회 학습\n"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(analysis_text)
            print(f"분석 결과 저장: {output_file}")
        except Exception as e:
            print(f"분석 결과 저장 실패: {e}")

    def cleanup(self):
        """리소스 정리"""
        self.save_learning_data()
        self.cleanup_old_data()
        self.save_learning_data()  # 정리 후 다시 저장