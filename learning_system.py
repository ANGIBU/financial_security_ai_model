# learning_system.py

import json
import pickle
import hashlib
import numpy as np
import re
import os
import tempfile
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import time

def _default_list():
    return []

def _default_int():
    return 0

def _default_float():
    return 0.0

def _default_int_dict():
    return defaultdict(_default_int)

def _default_counter():
    return Counter()

def atomic_save(obj, filepath: str) -> bool:
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

def atomic_load(filepath: str):
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None

@dataclass
class LearningData:
    question_id: str
    question: str
    correct_answer: str
    predicted_answer: str
    confidence: float
    is_correct: bool
    question_type: str
    domain: List[str]
    timestamp: float
    korean_quality: float = 0.0

class UnifiedLearningSystem:
    
    def __init__(self):
        self.training_data = []
        self.pattern_bank = defaultdict(_default_list)
        self.answer_statistics = defaultdict(_default_int_dict)
        
        self.korean_quality_stats = {
            "total_samples": 0,
            "high_quality": 0,
            "medium_quality": 0,
            "low_quality": 0,
            "avg_quality": 0.0
        }
        
        self.learned_rules = {}
        self.rule_performance = defaultdict(_default_list)
        
        self.korean_templates = self._initialize_korean_templates()
        self.successful_answers = defaultdict(_default_list)
        
        self.performance_history = []
        self.learning_metrics = {
            "total_samples": 0,
            "correct_predictions": 0,
            "patterns_learned": 0,
            "rules_created": 0,
            "korean_quality_avg": 0.0
        }
        
        self.prediction_cache = {}
        self.max_cache_size = 200
        
        self.word_frequency = defaultdict(_default_int)
        self.word_associations = defaultdict(_default_counter)
        self.context_patterns = defaultdict(_default_counter)
        self.domain_vocabularies = defaultdict(set)
        self.learned_sequences = defaultdict(_default_counter)
        
    def _initialize_korean_templates(self) -> Dict[str, List[str]]:
        return {
            "개인정보보호": [
                "개인정보보호법에 따라 {action}를 수행해야 합니다.",
                "정보주체의 권리 보호를 위해 {measure}가 필요합니다.",
                "개인정보의 안전한 관리를 위해 {requirement}가 요구됩니다.",
                "개인정보 처리방침을 수립하고 {action}를 이행해야 합니다.",
                "개인정보보호 관리체계를 구축하여 {measure}를 확보해야 합니다."
            ],
            "전자금융": [
                "전자금융거래법에 따라 {action}를 수행해야 합니다.",
                "전자적 장치를 통한 거래의 안전성 확보를 위해 {measure}가 필요합니다.",
                "접근매체 관리와 관련하여 {requirement}를 준수해야 합니다.",
                "전자금융거래의 신뢰성 향상을 위해 {action}가 요구됩니다.",
                "전자금융 보안대책을 수립하고 {measure}를 구현해야 합니다."
            ],
            "정보보안": [
                "정보보안 관리체계에 따라 {action}를 구현해야 합니다.",
                "체계적인 보안 관리를 위해 {measure}가 요구됩니다.",
                "위험평가를 통해 {requirement}를 수립해야 합니다.",
                "보안정책의 수립과 이행을 위해 {action}가 필요합니다.",
                "지속적인 보안 모니터링을 통해 {measure}를 확보해야 합니다."
            ],
            "암호화": [
                "중요 정보는 {action}를 통해 보호해야 합니다.",
                "암호화 기술을 활용하여 {measure}를 확보해야 합니다.",
                "안전한 키 관리를 위해 {requirement}가 필요합니다.",
                "전송 구간 암호화를 통해 {action}를 수행해야 합니다.",
                "저장 데이터 암호화를 위해 {measure}를 적용해야 합니다."
            ],
            "사고대응": [
                "{event} 발생 시 {action}를 수행해야 합니다.",
                "침해사고 대응은 {phase}별로 {measure}를 이행해야 합니다.",
                "복구 계획은 {target}을 고려하여 {requirement}를 수립해야 합니다.",
                "사고 대응팀 구성을 통해 {action}가 요구됩니다.",
                "신속한 복구를 위해 {measure}를 준비해야 합니다."
            ],
            "사이버보안": [
                "악성코드 탐지를 위해 {method}를 활용해야 합니다.",
                "트로이 목마는 {characteristic}를 가진 악성코드입니다.",
                "원격 접근 공격에 대비하여 {measure}가 필요합니다.",
                "시스템 감시를 통해 {indicator}를 확인해야 합니다.",
                "보안 솔루션을 통해 {action}를 수행해야 합니다."
            ],
            "위험관리": [
                "위험관리 체계를 수립하여 {action}를 수행해야 합니다.",
                "위험 식별과 평가를 통해 {measure}가 필요합니다.",
                "위험 대응 전략을 수립하고 {requirement}를 구현해야 합니다.",
                "지속적인 위험 모니터링을 통해 {action}를 강화해야 합니다.",
                "위험 관리 정책을 수립하여 {measure}를 확보해야 합니다."
            ],
            "관리체계": [
                "관리체계 수립을 위해 {action}가 필요합니다.",
                "정책 수립과 운영을 통해 {measure}를 확보해야 합니다.",
                "조직 구성과 역할 분담을 통해 {requirement}를 구현해야 합니다.",
                "정기적인 점검과 개선을 통해 {action}를 수행해야 합니다.",
                "관리체계의 지속적 개선을 위해 {measure}가 요구됩니다."
            ]
        }
    
    def _evaluate_korean_quality(self, text: str, question_type: str) -> float:
        if question_type == "multiple_choice":
            if re.match(r'^[1-5]$', text.strip()):
                return 1.0
            return 0.5
        
        if re.search(r'[\u4e00-\u9fff]', text):
            return 0.0
        
        if re.search(r'[а-яё]', text.lower()):
            return 0.0
        
        weird_chars = re.findall(r'[^\w\s가-힣0-9.,!?()·\-]', text)
        if len(weird_chars) > 5:
            return 0.1
        
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(re.sub(r'[^\w]', '', text))
        
        if total_chars == 0:
            return 0.0
        
        korean_ratio = korean_chars / total_chars
        english_chars = len(re.findall(r'[A-Za-z]', text))
        english_ratio = english_chars / total_chars
        
        professional_terms = ['법', '규정', '조치', '관리', '보안', '체계', '정책', '절차', '방안', '시스템', '대책']
        prof_count = sum(1 for term in professional_terms if term in text)
        prof_bonus = min(prof_count * 0.06, 0.25)
        
        quality = korean_ratio * 0.9
        quality -= english_ratio * 0.2
        quality += prof_bonus
        quality = max(0, min(1, quality))
        
        return quality
    
    def add_training_sample(self, question: str, correct_answer: str, 
                          predicted_answer: str, confidence: float,
                          question_type: str, domain: List[str],
                          question_id: str = None) -> None:
        
        is_correct = (correct_answer == predicted_answer)
        korean_quality = self._evaluate_korean_quality(predicted_answer, question_type)
        
        if korean_quality < 0.2 and question_type != "multiple_choice":
            return
        
        sample = LearningData(
            question_id=question_id or hashlib.md5(question.encode()).hexdigest()[:8],
            question=question,
            correct_answer=correct_answer,
            predicted_answer=predicted_answer,
            confidence=confidence,
            is_correct=is_correct,
            question_type=question_type,
            domain=domain,
            timestamp=time.time(),
            korean_quality=korean_quality
        )
        
        self.training_data.append(sample)
        if len(self.training_data) > 1500:
            self.training_data = self.training_data[-1500:]
        
        self.learning_metrics["total_samples"] += 1
        
        if is_correct:
            self.learning_metrics["correct_predictions"] += 1
        
        self._update_korean_quality_stats(korean_quality)
        
        if korean_quality > 0.5 and question_type != "multiple_choice" and confidence > 0.5:
            for d in domain:
                self.successful_answers[d].append({
                    "answer": predicted_answer,
                    "confidence": confidence,
                    "structure": self._analyze_answer_structure(predicted_answer)
                })
                if len(self.successful_answers[d]) > 30:
                    self.successful_answers[d] = self.successful_answers[d][-30:]
        
        self._extract_patterns(sample)
        self._learn_word_patterns(sample)
    
    def _learn_word_patterns(self, sample: LearningData):
        words = re.findall(r'[가-힣]{2,}', sample.question.lower())
        
        for word in words:
            self.word_frequency[word] += 1
            for domain in sample.domain:
                self.domain_vocabularies[domain].add(word)
        
        for i in range(len(words)):
            word = words[i]
            self.word_associations[word][sample.correct_answer] += sample.confidence
            
            for j in range(i+1, min(i+4, len(words))):
                context_window = " ".join(words[i:j+1])
                if len(context_window) <= 50:
                    self.context_patterns[context_window][sample.correct_answer] += sample.confidence
        
        sequences = []
        if "해당하지" in sample.question.lower():
            sequences.append("부정형")
        if "정의" in sample.question.lower():
            sequences.append("정의질문")
        if "방안" in sample.question.lower() or "대책" in sample.question.lower():
            sequences.append("방안질문")
        
        for seq in sequences:
            self.learned_sequences[seq][sample.correct_answer] += sample.confidence
        
        for domain in sample.domain:
            domain_words = list(self.domain_vocabularies[domain])
            question_words = set(words)
            overlap = len(question_words & set(domain_words))
            if overlap >= 2:
                domain_pattern = f"domain_{domain}_{overlap}"
                self.learned_sequences[domain_pattern][sample.correct_answer] += sample.confidence * 1.2
    
    def _update_korean_quality_stats(self, quality: float) -> None:
        self.korean_quality_stats["total_samples"] += 1
        
        if quality > 0.7:
            self.korean_quality_stats["high_quality"] += 1
        elif quality > 0.4:
            self.korean_quality_stats["medium_quality"] += 1
        else:
            self.korean_quality_stats["low_quality"] += 1
        
        total = self.korean_quality_stats["total_samples"]
        if total > 0:
            prev_avg = self.korean_quality_stats["avg_quality"]
            self.korean_quality_stats["avg_quality"] = (prev_avg * (total - 1) + quality) / total
            self.learning_metrics["korean_quality_avg"] = self.korean_quality_stats["avg_quality"]
    
    def _analyze_answer_structure(self, answer: str) -> Dict:
        return {
            "has_numbering": bool(re.search(r'첫째|둘째|1\)|2\)', answer)),
            "has_law_reference": bool(re.search(r'법|규정|조항', answer)),
            "has_conclusion": bool(re.search(r'따라서|그러므로|결론', answer)),
            "has_examples": bool(re.search(r'예를 들어|예시|구체적으로', answer)),
            "sentence_count": len(re.split(r'[.!?]', answer)),
            "length": len(answer)
        }
    
    def _extract_patterns(self, sample: LearningData) -> None:
        question_lower = sample.question.lower()
        
        keywords = self._extract_keywords_enhanced(question_lower)
        for keyword in keywords:
            self.pattern_bank[keyword].append({
                "answer": sample.correct_answer,
                "confidence": sample.confidence if sample.is_correct else sample.confidence * 0.4,
                "domain": sample.domain,
                "korean_quality": sample.korean_quality
            })
            
            if len(self.pattern_bank[keyword]) > 80:
                self.pattern_bank[keyword] = sorted(
                    self.pattern_bank[keyword], 
                    key=lambda x: x["confidence"], 
                    reverse=True
                )[:80]
        
        for domain in sample.domain:
            if domain not in self.answer_statistics:
                self.answer_statistics[domain] = defaultdict(_default_int)
            self.answer_statistics[domain][sample.correct_answer] += 1
        
        if sample.question_type == "multiple_choice":
            self._learn_mc_pattern(sample)
        else:
            self._learn_subjective_pattern(sample)
    
    def _extract_keywords_enhanced(self, text: str) -> List[str]:
        keywords = []
        
        key_terms = [
            "개인정보", "전자금융", "암호화", "보안", "관리체계", "위험관리", 
            "재해복구", "정보보호", "ISMS", "ISO", "접근매체", "안전성확보조치",
            "트로이", "악성코드", "피싱", "스미싱", "해킹", "방화벽", "백업",
            "취약점", "침입탐지", "침입방지", "다중인증", "생체인증", "소셜엔지니어링",
            "전자서명", "신용정보", "금융실명", "보험업법", "자본시장법", "은행법",
            "정보통신망법", "개인정보보호법", "전자금융거래법", "GDPR", "CCPA"
        ]
        for term in key_terms:
            if term in text:
                keywords.append(term)
        
        phrases = [
            "해당하지 않는", "적절하지 않은", "옳지 않은", "가장 적절한",
            "가장 중요한", "우선적으로", "반드시", "필수적으로"
        ]
        for phrase in phrases:
            if phrase in text:
                keywords.append(phrase.replace(" ", "_"))
        
        if "해당하지" in text or "적절하지" in text or "옳지않" in text:
            keywords.append("부정형")
        if "정의" in text or "의미" in text:
            keywords.append("정의문제")
        if "방안" in text or "대책" in text:
            keywords.append("방안문제")
        
        words = re.findall(r'[가-힣]{3,}', text)
        frequent_words = [word for word in words if self.word_frequency.get(word, 0) >= 3]
        keywords.extend(frequent_words[:5])
        
        return keywords
    
    def _learn_mc_pattern(self, sample: LearningData) -> None:
        if "해당하지" in sample.question or "적절하지" in sample.question:
            pattern_key = "negative_mc"
            if pattern_key not in self.learned_rules:
                self.learned_rules[pattern_key] = {
                    "answers": defaultdict(_default_float),
                    "confidence": 0.5,
                    "success_rate": 0.0,
                    "sample_count": 0
                }
            
            rule = self.learned_rules[pattern_key]
            rule["answers"][sample.correct_answer] += 1
            rule["sample_count"] += 1
            
            if sample.is_correct:
                rule["success_rate"] = (rule["success_rate"] * (rule["sample_count"] - 1) + 1) / rule["sample_count"]
            else:
                rule["success_rate"] = (rule["success_rate"] * (rule["sample_count"] - 1)) / rule["sample_count"]
            
            self.rule_performance[pattern_key].append(sample.is_correct)
    
    def _learn_subjective_pattern(self, sample: LearningData) -> None:
        if sample.korean_quality < 0.5:
            return
        
        for domain in sample.domain:
            pattern_key = f"subj_{domain}"
            if pattern_key not in self.learned_rules:
                self.learned_rules[pattern_key] = {
                    "templates": [],
                    "keywords": [],
                    "avg_quality": 0.0,
                    "success_rate": 0.0,
                    "sample_count": 0
                }
            
            rule = self.learned_rules[pattern_key]
            
            if sample.is_correct and len(sample.correct_answer) > 30:
                rule["templates"].append({
                    "text": sample.correct_answer[:300],
                    "quality": sample.korean_quality,
                    "structure": self._analyze_answer_structure(sample.correct_answer)
                })
                
                if len(rule["templates"]) > 15:
                    rule["templates"].sort(key=lambda x: x["quality"], reverse=True)
                    rule["templates"] = rule["templates"][:15]
                
                rule["sample_count"] += 1
                if sample.is_correct:
                    rule["success_rate"] = (rule["success_rate"] * (rule["sample_count"] - 1) + 1) / rule["sample_count"]
                else:
                    rule["success_rate"] = (rule["success_rate"] * (rule["sample_count"] - 1)) / rule["sample_count"]
    
    def predict_with_learning(self, question: str, question_type: str,
                            domain: List[str]) -> Tuple[str, float]:
        
        cache_key = hashlib.md5(f"{question}{question_type}".encode()).hexdigest()[:12]
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]
        
        if question_type == "multiple_choice":
            prediction = self._predict_mc_enhanced(question, domain)
        else:
            prediction = self._predict_subjective_korean(question, domain)
        
        if len(self.prediction_cache) >= self.max_cache_size:
            oldest_key = next(iter(self.prediction_cache))
            del self.prediction_cache[oldest_key]
        
        self.prediction_cache[cache_key] = prediction
        
        return prediction
    
    def _predict_mc_enhanced(self, question: str, domain: List[str]) -> Tuple[str, float]:
        question_lower = question.lower()
        prediction_scores = defaultdict(float)
        
        if "해당하지" in question or "적절하지" in question:
            if "negative_mc" in self.learned_rules:
                rule = self.learned_rules["negative_mc"]
                if rule["sample_count"] >= 2 and rule["success_rate"] > 0.3:
                    answers = rule["answers"]
                    if answers:
                        best_answer = max(answers.items(), key=lambda x: x[1])
                        confidence = min(best_answer[1] / sum(answers.values()) * rule["success_rate"], 0.85)
                        prediction_scores[best_answer[0]] += confidence
        
        words = re.findall(r'[가-힣]{2,}', question_lower)
        for word in words:
            if word in self.word_associations:
                total_count = sum(self.word_associations[word].values())
                if total_count >= 2:
                    for answer, count in self.word_associations[word].items():
                        weight = (count / total_count) * 0.4
                        prediction_scores[answer] += weight
        
        for i in range(len(words)-1):
            context = f"{words[i]} {words[i+1]}"
            if context in self.context_patterns:
                total_count = sum(self.context_patterns[context].values())
                if total_count >= 2:
                    for answer, count in self.context_patterns[context].items():
                        weight = (count / total_count) * 0.6
                        prediction_scores[answer] += weight
        
        for seq_key, answer_counts in self.learned_sequences.items():
            if ("부정형" in seq_key and ("해당하지" in question or "적절하지" in question)) or \
               ("정의질문" in seq_key and "정의" in question) or \
               ("방안질문" in seq_key and ("방안" in question or "대책" in question)):
                total_count = sum(answer_counts.values())
                if total_count >= 2:
                    for answer, count in answer_counts.items():
                        weight = (count / total_count) * 0.7
                        prediction_scores[answer] += weight
        
        for d in domain:
            if d in self.answer_statistics:
                stats = self.answer_statistics[d]
                if stats:
                    total = sum(stats.values())
                    if total >= 2:
                        for answer, count in stats.items():
                            weight = (count / total) * 0.5
                            prediction_scores[answer] += weight
        
        if prediction_scores:
            best_answer = max(prediction_scores.items(), key=lambda x: x[1])
            confidence = min(best_answer[1], 0.9)
            return best_answer[0], confidence
        
        return "3", 0.25
    
    def _predict_subjective_korean(self, question: str, domain: List[str]) -> Tuple[str, float]:
        for d in domain:
            if d in self.successful_answers and self.successful_answers[d]:
                candidates = sorted(
                    self.successful_answers[d],
                    key=lambda x: x["confidence"],
                    reverse=True
                )[:3]
                
                if candidates:
                    selected = candidates[0]
                    return selected["answer"], selected["confidence"] * 0.8
        
        for d in domain:
            pattern_key = f"subj_{d}"
            if pattern_key in self.learned_rules:
                rule = self.learned_rules[pattern_key]
                if rule.get("sample_count", 0) >= 1 and rule.get("success_rate", 0) > 0.4:
                    templates = rule.get("templates", [])
                    if templates:
                        best_template = max(templates, key=lambda x: x["quality"])
                        return best_template["text"], 0.6
        
        words = re.findall(r'[가-힣]{2,}', question.lower())
        for word in words:
            if word in self.word_associations:
                answer_counts = self.word_associations[word]
                total_count = sum(answer_counts.values())
                if total_count >= 3:
                    non_mc_answers = {k: v for k, v in answer_counts.items() 
                                    if not (k.isdigit() and 1 <= int(k) <= 5)}
                    if non_mc_answers:
                        best_answer = max(non_mc_answers.items(), key=lambda x: x[1])
                        confidence = best_answer[1] / total_count
                        if confidence > 0.4:
                            return best_answer[0], confidence * 0.7
        
        return self._generate_korean_fallback(domain), 0.4
    
    def _generate_korean_fallback(self, domains: List[str]) -> str:
        if "개인정보보호" in domains:
            return "개인정보보호법에 따라 개인정보의 수집과 이용 시 정보주체의 동의를 받아야 하며, 안전성 확보조치를 통해 개인정보를 보호해야 합니다."
        elif "전자금융" in domains:
            return "전자금융거래법에 따라 전자적 장치를 통한 금융거래의 안전성과 신뢰성을 확보해야 합니다."
        elif "정보보안" in domains:
            return "정보보안 관리체계를 구축하여 조직의 정보자산을 체계적으로 보호해야 합니다."
        elif "암호화" in domains:
            return "중요 정보는 안전한 암호 알고리즘을 사용하여 암호화해야 합니다."
        elif "사이버보안" in domains:
            return "트로이 목마는 정상 프로그램으로 위장한 악성코드로, 원격 접근 트로이 목마는 공격자가 감염된 시스템을 원격으로 제어할 수 있게 합니다. 주요 탐지 지표로는 비정상적인 네트워크 연결, 시스템 리소스 사용 증가, 알 수 없는 프로세스 실행 등이 있습니다."
        else:
            return "관련 법령과 규정에 따라 체계적인 보안 관리 방안을 수립하고 지속적인 개선을 수행해야 합니다."
    
    def get_current_accuracy(self) -> float:
        if self.learning_metrics["total_samples"] == 0:
            return 0.0
        
        return self.learning_metrics["correct_predictions"] / self.learning_metrics["total_samples"]
    
    def optimize_rules(self) -> None:
        rules_to_remove = []
        
        for rule_name, performance in self.rule_performance.items():
            if len(performance) >= 5:
                accuracy = sum(performance) / len(performance)
                if accuracy < 0.25:
                    rules_to_remove.append(rule_name)
        
        for rule in rules_to_remove:
            if rule in self.learned_rules:
                del self.learned_rules[rule]
            if rule in self.rule_performance:
                del self.rule_performance[rule]
        
        self._optimize_word_patterns()
        
        self.learning_metrics["rules_created"] = len(self.learned_rules)
    
    def _optimize_word_patterns(self):
        for word in list(self.word_associations.keys()):
            total_count = sum(self.word_associations[word].values())
            if total_count < 2:
                del self.word_associations[word]
            elif total_count > 50:
                top_answers = dict(Counter(self.word_associations[word]).most_common(5))
                self.word_associations[word] = Counter(top_answers)
        
        for context in list(self.context_patterns.keys()):
            total_count = sum(self.context_patterns[context].values())
            if total_count < 2:
                del self.context_patterns[context]
        
        for seq_key in list(self.learned_sequences.keys()):
            total_count = sum(self.learned_sequences[seq_key].values())
            if total_count < 2:
                del self.learned_sequences[seq_key]
    
    def save_learning_data(self, filepath: str = "./learning_data.pkl") -> bool:
        save_data = {
            "training_data": [asdict(d) for d in self.training_data[-500:]],
            "pattern_bank": {k: v[:30] for k, v in self.pattern_bank.items()},
            "answer_statistics": {k: dict(v) for k, v in self.answer_statistics.items()},
            "learned_rules": self.learned_rules,
            "rule_performance": {k: list(v)[-30:] for k, v in self.rule_performance.items()},
            "learning_metrics": self.learning_metrics,
            "korean_quality_stats": self.korean_quality_stats,
            "successful_answers": {k: v[-15:] for k, v in self.successful_answers.items()},
            "word_frequency": dict(self.word_frequency),
            "word_associations": {k: dict(v) for k, v in self.word_associations.items()},
            "context_patterns": {k: dict(v) for k, v in self.context_patterns.items()},
            "domain_vocabularies": {k: list(v) for k, v in self.domain_vocabularies.items()},
            "learned_sequences": {k: dict(v) for k, v in self.learned_sequences.items()}
        }
        
        return atomic_save(save_data, filepath)
    
    def load_learning_data(self, filepath: str = "./learning_data.pkl") -> bool:
        data = atomic_load(filepath)
        if data is None:
            return False
        
        try:
            self.training_data = [LearningData(**d) for d in data.get("training_data", [])]
            self.pattern_bank = defaultdict(_default_list, data.get("pattern_bank", {}))
            
            answer_stats = data.get("answer_statistics", {})
            self.answer_statistics = defaultdict(_default_int_dict)
            for k, v in answer_stats.items():
                self.answer_statistics[k] = defaultdict(_default_int, v)
            
            self.learned_rules = data.get("learned_rules", {})
            
            rule_perf = data.get("rule_performance", {})
            self.rule_performance = defaultdict(_default_list)
            for k, v in rule_perf.items():
                self.rule_performance[k] = list(v)
            
            self.learning_metrics = data.get("learning_metrics", self.learning_metrics)
            self.korean_quality_stats = data.get("korean_quality_stats", self.korean_quality_stats)
            
            success_answers = data.get("successful_answers", {})
            self.successful_answers = defaultdict(_default_list)
            for k, v in success_answers.items():
                self.successful_answers[k] = list(v)
            
            if "word_frequency" in data:
                self.word_frequency = defaultdict(_default_int, data["word_frequency"])
            
            if "word_associations" in data:
                word_assoc_data = data["word_associations"]
                self.word_associations = defaultdict(_default_counter)
                for k, v in word_assoc_data.items():
                    self.word_associations[k] = Counter(v)
            
            if "context_patterns" in data:
                context_data = data["context_patterns"]
                self.context_patterns = defaultdict(_default_counter)
                for k, v in context_data.items():
                    self.context_patterns[k] = Counter(v)
            
            if "domain_vocabularies" in data:
                domain_vocab_data = data["domain_vocabularies"]
                self.domain_vocabularies = defaultdict(set)
                for k, v in domain_vocab_data.items():
                    self.domain_vocabularies[k] = set(v)
            
            if "learned_sequences" in data:
                seq_data = data["learned_sequences"]
                self.learned_sequences = defaultdict(_default_counter)
                for k, v in seq_data.items():
                    self.learned_sequences[k] = Counter(v)
            
            return True
        except Exception:
            return False
    
    def get_learning_report(self) -> Dict:
        word_patterns_count = len(self.word_associations) + len(self.context_patterns) + len(self.learned_sequences)
        
        return {
            "total_samples": self.learning_metrics["total_samples"],
            "accuracy": self.get_current_accuracy(),
            "patterns_learned": word_patterns_count,
            "rules_created": self.learning_metrics["rules_created"],
            "korean_quality": {
                "average": self.korean_quality_stats["avg_quality"],
                "high_quality_rate": self.korean_quality_stats["high_quality"] / max(self.korean_quality_stats["total_samples"], 1),
                "low_quality_rate": self.korean_quality_stats["low_quality"] / max(self.korean_quality_stats["total_samples"], 1)
            },
            "domain_distribution": {
                domain: len(stats) 
                for domain, stats in self.answer_statistics.items()
            },
            "successful_templates": sum(len(answers) for answers in self.successful_answers.values()),
            "cache_size": len(self.prediction_cache),
            "word_patterns": {
                "word_associations": len(self.word_associations),
                "context_patterns": len(self.context_patterns),
                "learned_sequences": len(self.learned_sequences),
                "domain_vocabularies": sum(len(vocab) for vocab in self.domain_vocabularies.values())
            }
        }
    
    def cleanup(self):
        self.prediction_cache.clear()
        total = self.learning_metrics['total_samples']
        quality = self.korean_quality_stats['avg_quality']
        word_patterns = len(self.word_associations) + len(self.context_patterns)
        if total > 0:
            print(f"학습 시스템: {total}개 샘플, 품질 {quality:.2f}, 단어 패턴 {word_patterns}개")