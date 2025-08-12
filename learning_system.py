# learning_system.py

"""
학습 시스템
- 패턴 학습 및 예측
- 스마트 힌트 생성
- 자동 학습 및 교정
- 한국어 품질 관리
- 도메인별 답변 생성
"""

import re
import time
import torch
import numpy as np
import hashlib
import json
import pickle
import tempfile
import os
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter

def _default_int():
    return 0

def _default_float():
    return 0.0

def _default_list():
    return []

def _default_counter():
    return Counter()

def _default_float_dict():
    return defaultdict(_default_float)

def _default_int_dict():
    return defaultdict(_default_int)

def atomic_save_model(obj, filepath: str) -> bool:
    try:
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        fd, temp_path = tempfile.mkstemp(dir=directory)
        try:
            with os.fdopen(fd, 'wb') as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temp_path, filepath)
            return True
        except Exception:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise
    except Exception:
        return False

def atomic_load_model(filepath: str):
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None

@dataclass
class QuestionDifficulty:
    score: float
    factors: Dict[str, float]
    recommended_time: float
    recommended_attempts: int
    processing_priority: int
    memory_requirement: str

class LearningSystem:
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        
        self.pattern_weights = defaultdict(_default_float_dict)
        self.pattern_counts = defaultdict(_default_int)
        self.answer_distribution = {
            "mc": defaultdict(_default_int),
            "domain": defaultdict(_default_int_dict),
            "negative": defaultdict(_default_int)
        }
        
        self.learned_rules = self._initialize_rules()
        self.korean_templates = self._initialize_korean_templates()
        self.successful_answers = defaultdict(_default_list)
        
        self.learning_rate = 0.3
        self.confidence_threshold = 0.4
        self.min_samples = 1
        
        self.learning_history = []
        self.prediction_cache = {}
        self.pattern_cache = {}
        self.max_cache_size = 300
        
        self.stats = {
            "total_samples": 0,
            "correct_predictions": 0,
            "patterns_learned": 0,
            "korean_quality_avg": 0.0,
            "answer_diversity_score": 0.0
        }
        
        self.answer_patterns = self._initialize_balanced_patterns()
        self.answer_diversity_tracker = defaultdict(_default_int)
        
        if torch.cuda.is_available():
            self.gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        else:
            self.gpu_memory_total = 0
    
    def _debug_print(self, message: str):
        if self.debug_mode:
            print(f"[DEBUG] {message}")
    
    def _initialize_balanced_patterns(self) -> Dict:
        return {
            "금융투자업_분류": {
                "patterns": ["금융투자업", "구분", "해당하지", "소비자금융업", "투자매매업", "투자중개업", "보험중개업"],
                "preferred_answers": {"1": 0.25, "3": 0.22, "4": 0.20, "5": 0.18, "2": 0.15},
                "confidence": 0.65,
                "context_multipliers": {"소비자금융업": 1.2, "보험중개업": 1.1}
            },
            "위험관리_계획": {
                "patterns": ["위험", "관리", "계획", "수립", "고려", "요소", "적절하지"],
                "preferred_answers": {"3": 0.25, "1": 0.22, "4": 0.20, "2": 0.18, "5": 0.15},
                "confidence": 0.60,
                "context_multipliers": {"위험수용": 1.1, "적절하지": 1.05}
            },
            "관리체계_정책수립": {
                "patterns": ["관리체계", "수립", "운영", "정책수립", "단계", "중요한", "경영진"],
                "preferred_answers": {"1": 0.25, "3": 0.22, "2": 0.18, "4": 0.18, "5": 0.17},
                "confidence": 0.55,
                "context_multipliers": {"경영진": 1.1, "참여": 1.05}
            },
            "개인정보_정의": {
                "patterns": ["개인정보", "정의", "의미", "개념", "식별", "살아있는"],
                "preferred_answers": {"1": 0.24, "2": 0.22, "3": 0.20, "4": 0.17, "5": 0.17},
                "confidence": 0.60,
                "context_multipliers": {"개인정보보호법": 1.1}
            },
            "전자금융_정의": {
                "patterns": ["전자금융거래", "전자적장치", "금융상품", "서비스", "제공"],
                "preferred_answers": {"1": 0.22, "2": 0.21, "3": 0.20, "4": 0.19, "5": 0.18},
                "confidence": 0.55,
                "context_multipliers": {"전자금융거래법": 1.05}
            },
            "부정형_일반": {
                "patterns": ["해당하지", "적절하지", "옳지", "틀린", "잘못된"],
                "preferred_answers": {"1": 0.22, "3": 0.21, "4": 0.20, "5": 0.19, "2": 0.18},
                "confidence": 0.50,
                "context_multipliers": {"아닌": 1.05}
            }
        }
    
    def _initialize_rules(self) -> Dict:
        return {
            "개인정보_정의": {
                "keywords": ["개인정보", "정의", "의미", "개념", "식별"],
                "preferred_answers": {"1": 0.24, "2": 0.22, "3": 0.20, "4": 0.17, "5": 0.17},
                "confidence": 0.60
            },
            "전자금융_정의": {
                "keywords": ["전자금융", "정의", "전자적", "거래", "장치"],
                "preferred_answers": {"1": 0.22, "2": 0.21, "3": 0.20, "4": 0.19, "5": 0.18},
                "confidence": 0.55
            },
            "금융투자업_분류": {
                "keywords": ["금융투자업", "구분", "해당하지", "소비자금융", "보험중개"],
                "preferred_answers": {"1": 0.25, "3": 0.22, "4": 0.20, "5": 0.18, "2": 0.15},
                "confidence": 0.65
            },
            "위험관리_계획": {
                "keywords": ["위험", "관리", "계획", "수립", "고려", "요소"],
                "preferred_answers": {"3": 0.25, "1": 0.22, "4": 0.20, "2": 0.18, "5": 0.15},
                "confidence": 0.58
            }
        }
    
    def _initialize_korean_templates(self) -> Dict[str, List[str]]:
        return {
            "개인정보보호": [
                "개인정보보호법에 따라 개인정보의 안전한 관리와 정보주체의 권리 보호를 위한 체계적인 조치가 필요합니다.",
                "개인정보 처리 시 수집, 이용, 제공의 최소화 원칙을 준수하고 안전성 확보조치를 통해 보호해야 합니다.",
                "정보주체의 동의를 받아 개인정보를 수집하고, 목적 달성 후 지체 없이 파기해야 합니다.",
                "개인정보 처리방침을 수립하고 정보주체의 열람, 정정, 삭제 요구권을 보장해야 합니다.",
                "민감정보와 고유식별정보는 별도의 동의를 받아 처리하며 엄격한 보안조치를 적용해야 합니다."
            ],
            "전자금융": [
                "전자금융거래법에 따라 전자적 장치를 통한 금융거래의 안전성을 확보하고 이용자를 보호해야 합니다.",
                "접근매체의 안전한 관리와 거래내역 통지, 오류정정 절차를 구축해야 합니다.",
                "전자금융거래의 신뢰성 보장을 위해 적절한 보안조치와 이용자 보호 체계가 필요합니다.",
                "전자서명과 전자인증서를 통해 거래 당사자의 신원을 확인하고 거래의 무결성을 보장해야 합니다.",
                "오류 발생 시 신속한 정정 절차와 손해배상 체계를 마련하여 이용자 보호에 만전을 기해야 합니다."
            ],
            "정보보안": [
                "정보보안 관리체계를 통해 체계적인 보안 관리와 지속적인 위험 평가를 수행해야 합니다.",
                "정보자산의 기밀성, 무결성, 가용성을 보장하기 위한 종합적인 보안대책이 필요합니다.",
                "보안정책 수립, 접근통제, 암호화 등 다층적 보안체계를 구축해야 합니다.",
                "보안사고 예방과 대응을 위한 보안관제 체계와 침입탐지 시스템을 운영해야 합니다.",
                "정기적인 보안교육과 보안점검을 통해 보안 의식을 제고하고 취약점을 개선해야 합니다."
            ],
            "사이버보안": [
                "트로이 목마는 정상 프로그램으로 위장한 악성코드로, 시스템을 원격으로 제어할 수 있게 합니다.",
                "사이버 위협에 대응하기 위해 침입탐지, 방화벽, 보안관제 등 종합적 방어체계가 필요합니다.",
                "악성코드 탐지를 위해 실시간 모니터링과 행위 기반 분석 기술을 활용해야 합니다.",
                "피싱과 스미싱 등 사회공학 공격에 대한 사용자 교육과 기술적 차단 조치가 필요합니다.",
                "지능형 지속 위협에 대응하기 위해 위협 정보 공유와 협력 체계를 구축해야 합니다."
            ],
            "위험관리": [
                "위험관리는 조직의 목표 달성에 영향을 미칠 수 있는 위험을 체계적으로 식별하고 관리하는 과정입니다.",
                "위험 식별, 분석, 평가, 대응의 4단계 프로세스를 통해 체계적인 위험관리를 수행해야 합니다.",
                "위험 수용, 회피, 완화, 전가의 4가지 대응전략 중 적절한 방안을 선택하여 적용해야 합니다.",
                "정기적인 위험평가와 모니터링을 통해 위험 수준을 지속적으로 관리하고 개선해야 합니다.",
                "경영진의 위험관리 의지와 조직 전체의 위험 문화 조성이 성공적인 위험관리의 핵심입니다."
            ]
        }
    
    def evaluate_question_difficulty(self, question: str, structure: Dict) -> QuestionDifficulty:
        factors = {}
        
        length = len(question)
        factors["text_complexity"] = min(length / 1500, 0.2)
        
        choice_count = structure.get("choice_count", 0)
        factors["structural_complexity"] = min(choice_count / 8, 0.1)
        
        if structure.get("has_negative", False):
            factors["negative_complexity"] = 0.15
        else:
            factors["negative_complexity"] = 0.0
        
        tech_terms = len(structure.get("technical_terms", []))
        factors["technical_complexity"] = min(tech_terms / 5, 0.15)
        
        total_score = sum(factors.values())
        
        if total_score < 0.25:
            category = "fast"
            attempts = 1
            priority = 1
        elif total_score < 0.45:
            category = "normal"
            attempts = 2
            priority = 2
        else:
            category = "careful"
            attempts = 2
            priority = 3
        
        time_mapping = {
            "fast": 8,
            "normal": 15,
            "careful": 25
        }
        
        difficulty = QuestionDifficulty(
            score=total_score,
            factors=factors,
            recommended_time=time_mapping[category],
            recommended_attempts=attempts,
            processing_priority=priority,
            memory_requirement="medium"
        )
        
        return difficulty
    
    def get_smart_answer_hint(self, question: str, structure: Dict) -> Tuple[str, float]:
        question_id = hashlib.md5(question.encode()).hexdigest()[:8]
        
        if question_id in self.prediction_cache:
            return self.prediction_cache[question_id]
        
        question_normalized = re.sub(r'\s+', '', question.lower())
        
        total_distribution = dict(self.answer_diversity_tracker)
        total_answers = sum(total_distribution.values())
        
        if total_answers > 15:
            target_per_answer = total_answers / 5
            underrepresented = []
            for answer in ["1", "2", "3", "4", "5"]:
                current_count = total_distribution.get(answer, 0)
                if current_count < target_per_answer * 0.7:
                    underrepresented.append(answer)
            
            if underrepresented:
                selected = random.choice(underrepresented)
                result = (selected, 0.45)
                
                if len(self.prediction_cache) >= self.max_cache_size:
                    oldest_key = next(iter(self.prediction_cache))
                    del self.prediction_cache[oldest_key]
                self.prediction_cache[question_id] = result
                self.answer_diversity_tracker[result[0]] += 1
                return result
        
        best_match = None
        best_score = 0
        matched_rule_name = None
        
        for pattern_name, pattern_info in self.answer_patterns.items():
            patterns = pattern_info["patterns"]
            context_multipliers = pattern_info.get("context_multipliers", {})
            
            base_score = 0
            for pattern in patterns:
                if pattern.replace(" ", "") in question_normalized:
                    base_score += 1
            
            if base_score > 0:
                normalized_score = base_score / len(patterns)
                
                context_boost = 1.0
                for context, multiplier in context_multipliers.items():
                    if context.replace(" ", "") in question_normalized:
                        context_boost *= multiplier
                
                final_score = normalized_score * context_boost
                
                if final_score > best_score:
                    best_score = final_score
                    best_match = pattern_info
                    matched_rule_name = pattern_name
        
        if best_match and best_score > 0.25:
            answers = best_match["preferred_answers"]
            
            answer_options = []
            for answer, weight in answers.items():
                answer_options.extend([answer] * int(weight * 100))
            
            if answer_options:
                selected_answer = random.choice(answer_options)
                base_confidence = best_match["confidence"]
                adjusted_confidence = min(base_confidence * best_score * 0.8, 0.75)
                
                result = (selected_answer, adjusted_confidence)
            else:
                result = self._diversified_fallback(question, structure)
        else:
            result = self._diversified_fallback(question, structure)
        
        if len(self.prediction_cache) >= self.max_cache_size:
            oldest_key = next(iter(self.prediction_cache))
            del self.prediction_cache[oldest_key]
        self.prediction_cache[question_id] = result
        
        self.answer_diversity_tracker[result[0]] += 1
        
        return result
    
    def _diversified_fallback(self, question: str, structure: Dict) -> Tuple[str, float]:
        question_lower = question.lower()
        domains = structure.get("domain_hints", [])
        has_negative = structure.get("has_negative", False)
        
        question_hash = hash(question) % 100
        
        if has_negative:
            negative_options = ["1", "3", "4", "5"]
            if question_hash < 25:
                weights = [0.35, 0.25, 0.25, 0.15]
            elif question_hash < 50:
                weights = [0.25, 0.35, 0.25, 0.15]
            elif question_hash < 75:
                weights = [0.25, 0.25, 0.35, 0.15]
            else:
                weights = [0.25, 0.25, 0.25, 0.25]
            return random.choices(negative_options, weights=weights)[0], 0.50
        
        if "개인정보보호" in domains:
            if question_hash < 20:
                options = ["1", "2", "3", "4", "5"]
                weights = [0.3, 0.25, 0.2, 0.15, 0.1]
            elif question_hash < 40:
                options = ["2", "1", "3", "4", "5"]
                weights = [0.3, 0.25, 0.2, 0.15, 0.1]
            elif question_hash < 60:
                options = ["3", "1", "2", "4", "5"]
                weights = [0.3, 0.25, 0.2, 0.15, 0.1]
            elif question_hash < 80:
                options = ["4", "1", "2", "3", "5"]
                weights = [0.3, 0.25, 0.2, 0.15, 0.1]
            else:
                options = ["5", "1", "2", "3", "4"]
                weights = [0.3, 0.25, 0.2, 0.15, 0.1]
            return random.choices(options, weights=weights)[0], 0.45
        
        elif "전자금융" in domains:
            if question_hash < 25:
                base_answers = ["1", "2", "3"]
            elif question_hash < 50:
                base_answers = ["2", "3", "4"]
            elif question_hash < 75:
                base_answers = ["3", "4", "5"]
            else:
                base_answers = ["4", "5", "1"]
            return random.choice(base_answers), 0.45
        
        elif "정보보안" in domains or "사이버보안" in domains:
            if question_hash < 33:
                base_answers = ["1", "3", "4"]
            elif question_hash < 66:
                base_answers = ["2", "4", "5"]
            else:
                base_answers = ["3", "1", "5"]
            return random.choice(base_answers), 0.45
        
        if question_hash < 20:
            base_answers = ["1", "3", "4"]
        elif question_hash < 40:
            base_answers = ["2", "4", "5"]
        elif question_hash < 60:
            base_answers = ["3", "1", "5"]
        elif question_hash < 80:
            base_answers = ["4", "2", "1"]
        else:
            base_answers = ["5", "3", "2"]
        
        return random.choice(base_answers), 0.40
    
    def _evaluate_korean_quality(self, text: str, question_type: str) -> float:
        if question_type == "multiple_choice":
            if re.match(r'^[1-5]$', text.strip()):
                return 1.0
            return 0.5
        
        if not text or len(text) < 10:
            return 0.0
        
        if re.search(r'[\u4e00-\u9fff]', text):
            return 0.0
        
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(re.sub(r'[^\w]', '', text))
        
        if total_chars == 0:
            return 0.0
        
        korean_ratio = korean_chars / total_chars
        english_chars = len(re.findall(r'[A-Za-z]', text))
        english_ratio = english_chars / total_chars
        
        quality = korean_ratio * 0.8 - english_ratio * 0.1
        
        professional_terms = ['법', '규정', '조치', '관리', '보안', '체계', '정책']
        prof_count = sum(1 for term in professional_terms if term in text)
        quality += min(prof_count * 0.06, 0.2)
        
        if 30 <= len(text) <= 400:
            quality += 0.1
        
        return max(0, min(1, quality))
    
    def learn_from_prediction(self, question: str, prediction: str,
                            confidence: float, question_type: str,
                            domain: List[str]) -> None:
        
        if confidence < self.confidence_threshold:
            return
        
        korean_quality = self._evaluate_korean_quality(prediction, question_type)
        
        if korean_quality < 0.2 and question_type != "multiple_choice":
            return
        
        patterns = self._extract_patterns(question)
        
        for pattern in patterns:
            weight_boost = confidence * self.learning_rate * max(korean_quality, 0.3)
            self.pattern_weights[pattern][prediction] += weight_boost
            self.pattern_counts[pattern] += 1
        
        if question_type == "multiple_choice":
            self.answer_distribution["mc"][prediction] += 1
        
        for d in domain:
            if d not in self.answer_distribution["domain"]:
                self.answer_distribution["domain"][d] = defaultdict(_default_int)
            self.answer_distribution["domain"][d][prediction] += 1
        
        if korean_quality > 0.5 and question_type != "multiple_choice":
            self._learn_korean_patterns(prediction, domain)
        
        self.learning_history.append({
            "question_sample": question[:60],
            "prediction": prediction[:60] if len(prediction) > 60 else prediction,
            "confidence": confidence,
            "korean_quality": korean_quality,
            "patterns": len(patterns)
        })
        
        if len(self.learning_history) > 100:
            self.learning_history = self.learning_history[-100:]
        
        self.stats["total_samples"] += 1
        
        self._update_diversity_score()
    
    def _update_diversity_score(self):
        if len(self.answer_diversity_tracker) == 0:
            self.stats["answer_diversity_score"] = 0.0
            return
        
        total = sum(self.answer_diversity_tracker.values())
        if total == 0:
            self.stats["answer_diversity_score"] = 0.0
            return
        
        expected_ratio = 1.0 / 5.0
        chi_square = 0.0
        
        for i in range(1, 6):
            observed = self.answer_diversity_tracker.get(str(i), 0)
            expected = total * expected_ratio
            if expected > 0:
                chi_square += ((observed - expected) ** 2) / expected
        
        max_chi_square = 4 * total * expected_ratio
        diversity_score = 1.0 - (chi_square / max_chi_square) if max_chi_square > 0 else 0.0
        self.stats["answer_diversity_score"] = max(0.0, min(1.0, diversity_score))
    
    def _extract_patterns(self, question: str) -> List[str]:
        patterns = []
        question_lower = question.lower()
        
        for rule_name, rule_info in self.learned_rules.items():
            rule_keywords = rule_info["keywords"]
            match_count = sum(1 for keyword in rule_keywords if keyword in question_lower)
            
            if match_count >= 1:
                patterns.append(rule_name)
        
        if any(neg in question_lower for neg in ["해당하지", "적절하지", "옳지", "틀린"]):
            patterns.append("negative_question")
        
        if "정의" in question_lower:
            patterns.append("definition_question")
        if "방안" in question_lower or "대책" in question_lower:
            patterns.append("solution_question")
        
        return patterns[:8]
    
    def _learn_korean_patterns(self, text: str, domains: List[str]) -> None:
        if 30 <= len(text) <= 500:
            for domain in domains:
                self.successful_answers[domain].append({
                    "text": text,
                    "domains": domains,
                    "quality": self._evaluate_korean_quality(text, "subjective")
                })
                
                if len(self.successful_answers[domain]) > 30:
                    self.successful_answers[domain] = sorted(
                        self.successful_answers[domain],
                        key=lambda x: x["quality"],
                        reverse=True
                    )[:30]
    
    def predict_with_patterns(self, question: str, question_type: str) -> Tuple[str, float]:
        patterns = self._extract_patterns(question)
        
        if not patterns:
            return self._get_default_answer(question_type), 0.3
        
        for rule_name in patterns:
            if rule_name in self.learned_rules:
                rule = self.learned_rules[rule_name]
                answers = rule["preferred_answers"]
                
                answer_options = []
                for answer, weight in answers.items():
                    answer_options.extend([answer] * int(weight * 50))
                
                if answer_options:
                    selected = random.choice(answer_options)
                    return selected, rule["confidence"]
        
        answer_scores = defaultdict(_default_float)
        total_weight = 0
        
        for pattern in patterns:
            if pattern in self.pattern_weights and self.pattern_counts[pattern] >= self.min_samples:
                pattern_weight = self.pattern_counts[pattern]
                
                for answer, weight in self.pattern_weights[pattern].items():
                    answer_scores[answer] += weight * pattern_weight
                    total_weight += pattern_weight
        
        if not answer_scores:
            return self._get_default_answer(question_type), 0.3
        
        sorted_answers = sorted(answer_scores.items(), key=lambda x: x[1], reverse=True)
        top_answers = sorted_answers[:3]
        
        if top_answers:
            weights = [score for _, score in top_answers]
            total_weight = sum(weights)
            if total_weight > 0:
                normalized_weights = [w/total_weight for w in weights]
                selected_answer = random.choices([ans for ans, _ in top_answers], 
                                               weights=normalized_weights)[0]
                confidence = min(answer_scores[selected_answer] / max(total_weight, 1), 0.8)
                return selected_answer, confidence
        
        return self._get_default_answer(question_type), 0.3
    
    def _get_default_answer(self, question_type: str) -> str:
        if question_type == "multiple_choice":
            current_distribution = dict(self.answer_diversity_tracker)
            total = sum(current_distribution.values())
            
            if total > 10:
                underrepresented = []
                target_per_answer = total / 5
                for ans in ["1", "2", "3", "4", "5"]:
                    count = current_distribution.get(ans, 0)
                    if count < target_per_answer * 0.6:
                        underrepresented.append(ans)
                
                if underrepresented:
                    return random.choice(underrepresented)
            
            return random.choice(["1", "2", "3", "4", "5"])
        else:
            template_options = [
                "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 개선을 통해 안전성을 확보해야 합니다.",
                "정보보안 정책과 절차를 수립하여 체계적인 보안 관리와 위험 평가를 수행해야 합니다.",
                "적절한 기술적, 관리적, 물리적 보안조치를 통해 정보자산을 안전하게 보호해야 합니다.",
                "법령에서 요구하는 안전성 확보조치를 이행하고 정기적인 점검을 통해 개선해야 합니다.",
                "위험관리 체계를 구축하여 예방적 관리와 사후 대응 방안을 마련해야 합니다."
            ]
            return random.choice(template_options)
    
    def optimize_patterns(self) -> Dict:
        optimized = 0
        removed = 0
        
        patterns_to_remove = []
        for pattern, count in self.pattern_counts.items():
            if count < self.min_samples:
                patterns_to_remove.append(pattern)
        
        for pattern in patterns_to_remove:
            if pattern in self.pattern_weights:
                del self.pattern_weights[pattern]
            if pattern in self.pattern_counts:
                del self.pattern_counts[pattern]
            removed += 1
        
        for pattern in self.pattern_weights:
            total = sum(self.pattern_weights[pattern].values())
            if total > 0:
                answer_count = len(self.pattern_weights[pattern])
                if answer_count < 3:
                    for answer in ["1", "2", "3", "4", "5"]:
                        if answer not in self.pattern_weights[pattern]:
                            self.pattern_weights[pattern][answer] = total * 0.1
                optimized += 1
        
        self._update_diversity_score()
        
        return {
            "optimized": optimized,
            "removed": removed,
            "remaining": len(self.pattern_weights),
            "diversity_score": self.stats["answer_diversity_score"]
        }
    
    def get_current_accuracy(self) -> float:
        if self.stats["total_samples"] == 0:
            return 0.0
        return min(self.stats["correct_predictions"] / self.stats["total_samples"], 1.0)
    
    def save_model(self, filepath: str = "./learning_model.pkl") -> bool:
        model_data = {
            "pattern_weights": {k: dict(v) for k, v in self.pattern_weights.items()},
            "pattern_counts": dict(self.pattern_counts),
            "answer_distribution": {
                "mc": dict(self.answer_distribution["mc"]),
                "domain": {k: dict(v) for k, v in self.answer_distribution["domain"].items()},
                "negative": dict(self.answer_distribution["negative"])
            },
            "successful_answers": {k: v[-20:] for k, v in self.successful_answers.items()},
            "learning_history": self.learning_history[-50:],
            "learned_rules": self.learned_rules,
            "answer_diversity_tracker": dict(self.answer_diversity_tracker),
            "parameters": {
                "learning_rate": self.learning_rate,
                "confidence_threshold": self.confidence_threshold,
                "min_samples": self.min_samples
            }
        }
        
        return atomic_save_model(model_data, filepath)
    
    def load_model(self, filepath: str = "./learning_model.pkl") -> bool:
        model_data = atomic_load_model(filepath)
        if model_data is None:
            return False
        
        try:
            self.pattern_weights = defaultdict(_default_float_dict)
            for k, v in model_data.get("pattern_weights", {}).items():
                self.pattern_weights[k] = defaultdict(_default_float, v)
            
            self.pattern_counts = defaultdict(_default_int, model_data.get("pattern_counts", {}))
            
            answer_dist = model_data.get("answer_distribution", {})
            self.answer_distribution = {
                "mc": defaultdict(_default_int, answer_dist.get("mc", {})),
                "domain": defaultdict(_default_int_dict),
                "negative": defaultdict(_default_int, answer_dist.get("negative", {}))
            }
            
            for k, v in answer_dist.get("domain", {}).items():
                self.answer_distribution["domain"][k] = defaultdict(_default_int, v)
            
            self.successful_answers = defaultdict(_default_list, model_data.get("successful_answers", {}))
            self.learning_history = model_data.get("learning_history", [])
            self.answer_diversity_tracker = defaultdict(_default_int, model_data.get("answer_diversity_tracker", {}))
            
            if "learned_rules" in model_data:
                self.learned_rules.update(model_data["learned_rules"])
            
            params = model_data.get("parameters", {})
            self.learning_rate = params.get("learning_rate", 0.3)
            self.confidence_threshold = params.get("confidence_threshold", 0.4)
            self.min_samples = params.get("min_samples", 1)
            
            self._update_diversity_score()
            
            return True
        except Exception:
            return False
    
    def cleanup(self):
        total_patterns = len(self.pattern_weights)
        total_samples = len(self.learning_history)
        diversity = self.stats.get("answer_diversity_score", 0)
        if total_patterns > 0 or total_samples > 0:
            print(f"학습 시스템: {total_patterns}개 패턴, {total_samples}개 샘플, 다양성 {diversity:.2f}")