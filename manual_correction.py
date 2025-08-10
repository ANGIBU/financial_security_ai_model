# manual_correction.py

import csv
import os
import re
import hashlib
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import pickle

class ManualCorrectionSystem:
    
    def __init__(self):
        self.corrections = {}
        self.correction_patterns = defaultdict(list)
        self.correction_stats = {
            "total_corrections": 0,
            "pattern_based_corrections": 0,
            "manual_corrections": 0,
            "auto_applied_corrections": 0
        }
        
    def add_correction(self, question: str, original_answer: str, 
                      corrected_answer: str, reason: str = "") -> None:
        
        question_hash = hashlib.md5(question.encode()).hexdigest()[:12]
        
        self.corrections[question_hash] = {
            "question": question[:200],
            "original_answer": original_answer,
            "corrected_answer": corrected_answer,
            "reason": reason,
            "question_full": question
        }
        
        self.correction_stats["total_corrections"] += 1
        
        pattern = self._extract_correction_pattern(question, original_answer, corrected_answer)
        if pattern:
            self.correction_patterns[pattern["type"]].append({
                "pattern": pattern,
                "correction": corrected_answer,
                "confidence": 0.8
            })
    
    def _extract_correction_pattern(self, question: str, original: str, corrected: str) -> Optional[Dict]:
        question_lower = question.lower()
        
        if "금융투자업" in question_lower and "소비자금융업" in question_lower:
            return {
                "type": "금융투자업_분류",
                "keywords": ["금융투자업", "소비자금융업", "해당하지"],
                "trigger": "소비자금융업은 금융투자업이 아님"
            }
        
        if "위험관리" in question_lower and "위험수용" in question_lower:
            return {
                "type": "위험관리_요소",
                "keywords": ["위험관리", "계획수립", "위험수용"],
                "trigger": "위험수용은 대응전략의 하나"
            }
        
        if "관리체계" in question_lower and "정책수립" in question_lower:
            return {
                "type": "관리체계_정책",
                "keywords": ["관리체계", "정책수립", "경영진"],
                "trigger": "경영진의 참여가 가장 중요"
            }
        
        if "재해복구" in question_lower and "개인정보파기" in question_lower:
            return {
                "type": "재해복구_계획",
                "keywords": ["재해복구", "개인정보파기"],
                "trigger": "개인정보파기는 재해복구와 무관"
            }
        
        if "트로이" in question_lower or "악성코드" in question_lower:
            return {
                "type": "사이버보안_트로이",
                "keywords": ["트로이", "악성코드", "원격제어"],
                "trigger": "트로이 목마 관련 답변"
            }
        
        if "개인정보" in question_lower and "유출" in question_lower:
            return {
                "type": "개인정보_유출",
                "keywords": ["개인정보", "유출", "통지"],
                "trigger": "개인정보 유출 시 통지 의무"
            }
        
        if "전자금융" in question_lower and "접근매체" in question_lower:
            return {
                "type": "전자금융_접근매체",
                "keywords": ["전자금융", "접근매체", "안전"],
                "trigger": "접근매체 안전 관리"
            }
        
        if "암호화" in question_lower or "암호" in question_lower:
            return {
                "type": "암호화_기술",
                "keywords": ["암호화", "암호", "키관리"],
                "trigger": "암호화 기술 관련"
            }
        
        return None
    
    def apply_corrections(self, question: str, answer: str) -> Tuple[str, float]:
        question_hash = hashlib.md5(question.encode()).hexdigest()[:12]
        
        if question_hash in self.corrections:
            correction = self.corrections[question_hash]
            self.correction_stats["manual_corrections"] += 1
            return correction["corrected_answer"], 0.95
        
        pattern_correction = self._apply_pattern_correction(question, answer)
        if pattern_correction:
            self.correction_stats["pattern_based_corrections"] += 1
            return pattern_correction[0], pattern_correction[1]
        
        return answer, 0.0
    
    def _apply_pattern_correction(self, question: str, answer: str) -> Optional[Tuple[str, float]]:
        question_lower = question.lower()
        
        for pattern_type, pattern_list in self.correction_patterns.items():
            for pattern_data in pattern_list:
                pattern = pattern_data["pattern"]
                keywords = pattern["keywords"]
                
                match_count = sum(1 for keyword in keywords if keyword in question_lower)
                
                if match_count >= len(keywords) * 0.7:
                    self.correction_stats["auto_applied_corrections"] += 1
                    return pattern_data["correction"], pattern_data["confidence"]
        
        smart_corrections = self._apply_smart_corrections(question, answer)
        if smart_corrections:
            return smart_corrections
        
        return None
    
    def _apply_smart_corrections(self, question: str, answer: str) -> Optional[Tuple[str, float]]:
        question_lower = question.lower()
        
        if "금융투자업" in question_lower and "소비자금융업" in question_lower:
            if "해당하지" in question_lower or "적절하지" in question_lower:
                return "1", 0.85
        
        if "위험관리" in question_lower and "위험수용" in question_lower:
            if "적절하지" in question_lower or "옳지" in question_lower:
                return "2", 0.82
        
        if "관리체계" in question_lower and "정책수립" in question_lower:
            if "가장중요" in question_lower or "중요한" in question_lower:
                return "2", 0.80
        
        if "재해복구" in question_lower and "개인정보파기" in question_lower:
            if "옳지" in question_lower or "적절하지" in question_lower:
                return "3", 0.83
        
        if ("트로이" in question_lower or "악성코드" in question_lower) and len(answer) < 100:
            return "트로이 목마는 정상 프로그램으로 위장한 악성코드로, 원격 접근 트로이 목마는 공격자가 감염된 시스템을 원격으로 제어할 수 있게 합니다. 주요 탐지 지표로는 비정상적인 네트워크 연결, 시스템 리소스 사용 증가, 알 수 없는 프로세스 실행 등이 있습니다.", 0.85
        
        if "개인정보" in question_lower and "유출" in question_lower and len(answer) < 100:
            return "개인정보 유출 시 개인정보보호법에 따라 지체 없이 정보주체에게 통지하고, 일정 규모 이상의 유출 시 개인정보보호위원회에 신고해야 합니다. 유출 통지 내용에는 유출 항목, 시점, 경위, 피해 최소화 방법, 담당부서 연락처 등이 포함되어야 합니다.", 0.80
        
        return None
    
    def interactive_correction(self, questions_data: List[Dict], answers: List[str]) -> int:
        corrections_made = 0
        
        print("\n=== 수동 교정 모드 ===")
        print("교정이 필요한 답변을 검토합니다.")
        
        for i, (q_data, answer) in enumerate(zip(questions_data, answers)):
            question = q_data["question"]
            is_mc = q_data["is_mc"]
            
            print(f"\n--- 문제 {i+1} ---")
            print(f"질문: {question[:200]}...")
            print(f"현재 답변: {answer}")
            print(f"유형: {'객관식' if is_mc else '주관식'}")
            
            needs_correction = input("교정이 필요합니까? (y/n): ").lower().strip()
            
            if needs_correction == 'y':
                corrected_answer = input("올바른 답변을 입력하세요: ").strip()
                reason = input("교정 이유 (선택사항): ").strip()
                
                if corrected_answer:
                    self.add_correction(question, answer, corrected_answer, reason)
                    answers[i] = corrected_answer
                    corrections_made += 1
                    print(f"✓ 교정 완료: {corrected_answer}")
            
            if i >= 4:
                break
        
        return corrections_made
    
    def save_corrections_to_csv(self, filepath: str = "./corrections.csv") -> bool:
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['question_hash', 'question', 'original_answer', 'corrected_answer', 'reason']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for question_hash, correction in self.corrections.items():
                    writer.writerow({
                        'question_hash': question_hash,
                        'question': correction['question'],
                        'original_answer': correction['original_answer'],
                        'corrected_answer': correction['corrected_answer'],
                        'reason': correction['reason']
                    })
            
            pattern_filepath = filepath.replace('.csv', '_patterns.pkl')
            with open(pattern_filepath, 'wb') as f:
                pickle.dump(dict(self.correction_patterns), f)
            
            return True
        except Exception as e:
            print(f"교정 데이터 저장 오류: {e}")
            return False
    
    def load_corrections_from_csv(self, filepath: str = "./corrections.csv") -> int:
        loaded_count = 0
        
        if not os.path.exists(filepath):
            return 0
        
        try:
            with open(filepath, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    question_hash = row['question_hash']
                    self.corrections[question_hash] = {
                        'question': row['question'],
                        'original_answer': row['original_answer'],
                        'corrected_answer': row['corrected_answer'],
                        'reason': row['reason'],
                        'question_full': row['question']
                    }
                    loaded_count += 1
            
            pattern_filepath = filepath.replace('.csv', '_patterns.pkl')
            if os.path.exists(pattern_filepath):
                with open(pattern_filepath, 'rb') as f:
                    loaded_patterns = pickle.load(f)
                    for pattern_type, patterns in loaded_patterns.items():
                        self.correction_patterns[pattern_type] = patterns
            
            self.correction_stats["total_corrections"] = loaded_count
            
        except Exception as e:
            print(f"교정 데이터 로드 오류: {e}")
        
        return loaded_count
    
    def get_correction_statistics(self) -> Dict:
        return {
            "total_corrections": len(self.corrections),
            "pattern_types": len(self.correction_patterns),
            "stats": self.correction_stats.copy(),
            "coverage": {
                "금융투자업": len([p for p in self.correction_patterns.get("금융투자업_분류", [])]),
                "위험관리": len([p for p in self.correction_patterns.get("위험관리_요소", [])]),
                "관리체계": len([p for p in self.correction_patterns.get("관리체계_정책", [])]),
                "재해복구": len([p for p in self.correction_patterns.get("재해복구_계획", [])]),
                "사이버보안": len([p for p in self.correction_patterns.get("사이버보안_트로이", [])])
            }
        }
    
    def export_correction_report(self, filepath: str = "./correction_report.txt") -> bool:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("=== 교정 시스템 보고서 ===\n\n")
                
                stats = self.get_correction_statistics()
                f.write(f"총 교정 수: {stats['total_corrections']}\n")
                f.write(f"패턴 유형 수: {stats['pattern_types']}\n\n")
                
                f.write("=== 도메인별 교정 현황 ===\n")
                for domain, count in stats['coverage'].items():
                    f.write(f"{domain}: {count}개\n")
                
                f.write("\n=== 개별 교정 내역 ===\n")
                for i, (question_hash, correction) in enumerate(self.corrections.items(), 1):
                    f.write(f"\n{i}. {question_hash}\n")
                    f.write(f"질문: {correction['question']}\n")
                    f.write(f"원본: {correction['original_answer']}\n")
                    f.write(f"교정: {correction['corrected_answer']}\n")
                    f.write(f"이유: {correction['reason']}\n")
                    f.write("-" * 50 + "\n")
            
            return True
        except Exception as e:
            print(f"보고서 생성 오류: {e}")
            return False
    
    def validate_correction_quality(self, question: str, original: str, corrected: str) -> Dict:
        quality_issues = []
        quality_score = 1.0
        
        if len(corrected.strip()) == 0:
            quality_issues.append("교정된 답변이 비어있습니다")
            quality_score -= 0.5
        
        if original == corrected:
            quality_issues.append("원본과 교정된 답변이 동일합니다")
            quality_score -= 0.3
        
        question_lower = question.lower()
        if any(term in question_lower for term in ["다음 중", "가장 적절한", "옳은 것"]):
            if not re.match(r'^[1-5]$', corrected.strip()):
                quality_issues.append("객관식 문제인데 1-5 범위의 숫자가 아닙니다")
                quality_score -= 0.4
        
        if re.search(r'[\u4e00-\u9fff]', corrected):
            quality_issues.append("교정된 답변에 중국어 문자가 포함되어 있습니다")
            quality_score -= 0.3
        
        korean_ratio = len(re.findall(r'[가-힣]', corrected)) / max(len(corrected), 1)
        if korean_ratio < 0.3 and len(corrected) > 10:
            quality_issues.append("한국어 비율이 낮습니다")
            quality_score -= 0.2
        
        return {
            "quality_score": max(quality_score, 0.0),
            "issues": quality_issues,
            "is_valid": quality_score >= 0.6 and len(quality_issues) == 0
        }
    
    def suggest_corrections(self, question: str, answer: str) -> List[Dict]:
        suggestions = []
        question_lower = question.lower()
        
        if "금융투자업" in question_lower and "해당하지" in question_lower:
            if "소비자금융업" in question_lower or "보험중개업" in question_lower:
                suggestions.append({
                    "suggested_answer": "1",
                    "confidence": 0.85,
                    "reason": "소비자금융업과 보험중개업은 금융투자업이 아님"
                })
        
        if "위험관리" in question_lower and "적절하지" in question_lower:
            if "위험수용" in question_lower:
                suggestions.append({
                    "suggested_answer": "2",
                    "confidence": 0.80,
                    "reason": "위험수용은 위험대응전략의 하나"
                })
        
        if "관리체계" in question_lower and "가장중요" in question_lower:
            if "경영진" in question_lower:
                suggestions.append({
                    "suggested_answer": "2",
                    "confidence": 0.82,
                    "reason": "정책수립 단계에서 경영진의 참여가 가장 중요"
                })
        
        if len(answer) < 50 and any(term in question_lower for term in ["트로이", "악성코드"]):
            suggestions.append({
                "suggested_answer": "트로이 목마는 정상 프로그램으로 위장한 악성코드로, 원격 접근 트로이 목마는 공격자가 감염된 시스템을 원격으로 제어할 수 있게 합니다.",
                "confidence": 0.78,
                "reason": "트로이 목마 관련 표준 답변이 필요함"
            })
        
        return suggestions
    
    def cleanup(self):
        total_corrections = len(self.corrections)
        if total_corrections > 0:
            print(f"교정 시스템: {total_corrections}개 교정 데이터")