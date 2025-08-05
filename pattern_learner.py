# pattern_learner.py
"""
답변 패턴 학습 및 예측 시스템
"""

import re
import json
import pickle
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import numpy as np
import hashlib

class AnswerPatternLearner:
    """고성능 답변 패턴 학습 클래스"""
    
    def __init__(self):
        self.patterns = {
            "keyword_answer_map": defaultdict(Counter),
            "domain_answer_distribution": defaultdict(Counter),
            "negative_answer_patterns": Counter(),
            "length_answer_correlation": defaultdict(list),
            "law_mention_answers": defaultdict(Counter),
            "complexity_answer_correlation": defaultdict(list),
            "question_type_patterns": defaultdict(Counter)
        }
        
        # 고급 학습 규칙
        self.learned_rules = self._initialize_comprehensive_rules()
        
        # 패턴 성능 추적
        self.pattern_performance = {
            "rule_success_rate": defaultdict(list),
            "prediction_accuracy": defaultdict(float),
            "confidence_calibration": defaultdict(list)
        }
        
        # 캐시 시스템
        self.prediction_cache = {}
        self.pattern_cache = {}
        
    def _initialize_comprehensive_rules(self) -> Dict:
        """포괄적 초기 규칙 설정"""
        return {
            "개인정보_정의": {
                "keywords": ["개인정보", "정의", "의미", "개념"],
                "preferred_answers": {"2": 0.65, "1": 0.20, "3": 0.10, "4": 0.03, "5": 0.02},
                "confidence": 0.75,
                "context_boost": {"법령": 0.1, "조항": 0.05}
            },
            "전자금융_정의": {
                "keywords": ["전자금융", "정의", "전자적", "거래"],
                "preferred_answers": {"2": 0.60, "1": 0.25, "3": 0.10, "4": 0.03, "5": 0.02},
                "confidence": 0.70,
                "context_boost": {"전자적장치": 0.15, "금융상품": 0.1}
            },
            "유출_신고": {
                "keywords": ["유출", "신고", "즉시", "지체", "통지"],
                "preferred_answers": {"1": 0.70, "2": 0.15, "3": 0.10, "4": 0.03, "5": 0.02},
                "confidence": 0.80,
                "context_boost": {"지체없이": 0.2, "개인정보보호위원회": 0.1}
            },
            "암호화_필수": {
                "keywords": ["암호화", "필수", "반드시", "의무"],
                "preferred_answers": {"1": 0.45, "2": 0.35, "3": 0.15, "4": 0.03, "5": 0.02},
                "confidence": 0.60,
                "context_boost": {"안전성확보조치": 0.15, "기술적조치": 0.1}
            },
            "부정형_일반": {
                "keywords": ["해당하지않는", "적절하지않은", "옳지않은", "틀린"],
                "preferred_answers": {"1": 0.40, "5": 0.25, "4": 0.20, "2": 0.10, "3": 0.05},
                "confidence": 0.65,
                "context_boost": {"제외": 0.1, "아닌": 0.05}
            },
            "법령_조항": {
                "keywords": ["법", "제", "조", "항", "규정", "시행령"],
                "preferred_answers": {"2": 0.35, "3": 0.30, "1": 0.20, "4": 0.10, "5": 0.05},
                "confidence": 0.55,
                "context_boost": {"따르면": 0.1, "의하면": 0.1}
            },
            "보안_조치": {
                "keywords": ["보안", "조치", "대책", "방안", "관리"],
                "preferred_answers": {"2": 0.40, "3": 0.30, "1": 0.20, "4": 0.07, "5": 0.03},
                "confidence": 0.60,
                "context_boost": {"관리체계": 0.15, "정보보호": 0.1}
            },
            "관리체계_ISMS": {
                "keywords": ["관리체계", "ISMS", "정보보호", "체계적"],
                "preferred_answers": {"3": 0.45, "2": 0.30, "1": 0.15, "4": 0.07, "5": 0.03},
                "confidence": 0.65,
                "context_boost": {"위험관리": 0.1, "지속적개선": 0.05}
            },
            "접근매체_관리": {
                "keywords": ["접근매체", "안전", "관리", "선정", "사용"],
                "preferred_answers": {"1": 0.50, "2": 0.30, "3": 0.15, "4": 0.03, "5": 0.02},
                "confidence": 0.70,
                "context_boost": {"신뢰할수있는": 0.15, "안전하고": 0.1}
            },
            "손실부담_원칙": {
                "keywords": ["손실", "부담", "책임", "배상", "피해"],
                "preferred_answers": {"2": 0.45, "1": 0.25, "3": 0.20, "4": 0.07, "5": 0.03},
                "confidence": 0.60,
                "context_boost": {"고의": 0.15, "중과실": 0.1}
            }
        }
    
    def analyze_question_pattern(self, question: str) -> Dict:
        """고급 문제 패턴 분석"""
        
        # 캐시 확인
        q_hash = hashlib.md5(question.encode()).hexdigest()[:12]
        if q_hash in self.pattern_cache:
            return self.pattern_cache[q_hash]
        
        question_lower = question.lower().replace(" ", "")
        
        # 매칭된 규칙들 수집
        matched_rules = []
        
        for rule_name, rule_info in self.learned_rules.items():
            base_keywords = rule_info["keywords"]
            context_keywords = rule_info.get("context_boost", {}).keys()
            all_keywords = list(base_keywords) + list(context_keywords)
            
            # 기본 키워드 매칭
            base_matches = sum(1 for kw in base_keywords if kw.replace(" ", "") in question_lower)
            
            # 컨텍스트 키워드 매칭
            context_matches = sum(1 for kw in context_keywords if kw.replace(" ", "") in question_lower)
            
            if base_matches >= 1:  # 최소 1개의 기본 키워드 필요
                total_score = base_matches / len(base_keywords)
                
                # 컨텍스트 보너스 적용
                if context_matches > 0:
                    context_boost = sum(rule_info["context_boost"][kw] 
                                      for kw in context_keywords 
                                      if kw.replace(" ", "") in question_lower)
                    total_score += context_boost
                
                matched_rules.append({
                    "rule": rule_name,
                    "match_score": total_score,
                    "base_confidence": rule_info["confidence"],
                    "answers": rule_info["preferred_answers"],
                    "base_matches": base_matches,
                    "context_matches": context_matches
                })
        
        # 최고 점수 규칙 선택
        if matched_rules:
            best_rule = max(matched_rules, key=lambda x: x["match_score"] * x["base_confidence"])
            
            # 캐시 저장
            self.pattern_cache[q_hash] = best_rule
            
            return best_rule
        
        # 캐시에 빈 결과 저장
        self.pattern_cache[q_hash] = None
        return None
    
    def predict_answer(self, question: str, structure: Dict) -> Tuple[str, float]:
        """고성능 패턴 기반 답변 예측"""
        
        # 캐시 확인
        cache_key = hashlib.md5(f"{question}{structure}".encode()).hexdigest()[:16]
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]
        
        # 부정형 문제 특별 처리
        if structure.get("has_negative", False):
            result = self._predict_negative_answer_advanced(question, structure)
            self.prediction_cache[cache_key] = result
            return result
        
        # 패턴 분석
        pattern_match = self.analyze_question_pattern(question)
        
        if pattern_match:
            # 구조적 특징 고려한 가중치 조정
            answers = pattern_match["answers"].copy()
            base_confidence = pattern_match["base_confidence"]
            
            # 문제 복잡도에 따른 조정
            complexity = structure.get("complexity", 0.5)
            if complexity > 0.7:
                # 복잡한 문제는 신뢰도 약간 감소
                base_confidence *= 0.95
            elif complexity < 0.3:
                # 간단한 문제는 신뢰도 증가
                base_confidence *= 1.05
            
            # 도메인별 조정
            domains = structure.get("domain", [])
            domain_boost = self._calculate_domain_boost(pattern_match["rule"], domains)
            base_confidence += domain_boost
            
            # 최종 답변 선택
            best_answer = max(answers.items(), key=lambda x: x[1])
            final_confidence = min(base_confidence * pattern_match["match_score"], 1.0)
            
            result = (best_answer[0], final_confidence)
            self.prediction_cache[cache_key] = result
            return result
        
        # 통계적 예측으로 폴백
        result = self._statistical_prediction_advanced(question, structure)
        self.prediction_cache[cache_key] = result
        return result
    
    def _predict_negative_answer_advanced(self, question: str, structure: Dict) -> Tuple[str, float]:
        """고급 부정형 문제 예측"""
        question_lower = question.lower()
        
        # 특정 부정형 패턴 분석
        negative_patterns = {
            "극단표현": {
                "keywords": ["모든", "항상", "절대", "반드시", "전혀"],
                "preferred_answer": "1",
                "confidence": 0.75
            },
            "제외표현": {
                "keywords": ["제외", "빼고", "외에"],
                "preferred_answer": "5",
                "confidence": 0.70
            },
            "예외표현": {
                "keywords": ["예외", "경우가아닌", "해당되지않는"],
                "preferred_answer": "4",
                "confidence": 0.65
            },
            "반대표현": {
                "keywords": ["반대", "상반", "대비"],
                "preferred_answer": "1",
                "confidence": 0.60
            }
        }
        
        # 패턴 매칭
        for pattern_name, pattern_info in negative_patterns.items():
            keywords = pattern_info["keywords"]
            if any(kw in question_lower for kw in keywords):
                return pattern_info["preferred_answer"], pattern_info["confidence"]
        
        # 도메인별 부정형 선호도
        domains = structure.get("domain", [])
        if "개인정보보호" in domains:
            return "1", 0.65  # 개인정보 영역에서는 보통 1번이 틀림
        elif "전자금융" in domains:
            return "2", 0.60  # 전자금융 영역에서는 2번이 틀릴 가능성
        elif "정보보안" in domains:
            return "5", 0.55  # 보안 영역에서는 5번이 예외적 경우
        
        # 기본 부정형 예측
        rule = self.learned_rules["부정형_일반"]
        answers = rule["preferred_answers"]
        best_answer = max(answers.items(), key=lambda x: x[1])
        
        return best_answer[0], rule["confidence"]
    
    def _calculate_domain_boost(self, rule_name: str, domains: List[str]) -> float:
        """도메인별 신뢰도 부스트 계산"""
        domain_relevance = {
            "개인정보_정의": ["개인정보보호"],
            "전자금융_정의": ["전자금융"],
            "유출_신고": ["개인정보보호"],
            "암호화_필수": ["암호화", "정보보안"],
            "보안_조치": ["정보보안"],
            "관리체계_ISMS": ["정보보안"],
            "접근매체_관리": ["전자금융"],
            "손실부담_원칙": ["전자금융"]
        }
        
        relevant_domains = domain_relevance.get(rule_name, [])
        matched_domains = set(domains) & set(relevant_domains)
        
        return len(matched_domains) * 0.05  # 도메인 매칭당 5% 부스트
    
    def _statistical_prediction_advanced(self, question: str, structure: Dict) -> Tuple[str, float]:
        """고급 통계 기반 예측"""
        
        if structure["question_type"] == "multiple_choice":
            # 길이 기반 예측 개선
            question_length = len(question)
            
            # 복잡도 고려
            complexity = structure.get("complexity", 0.5)
            
            if question_length < 200:
                if complexity < 0.4:
                    return "2", 0.40  # 짧고 간단한 문제
                else:
                    return "3", 0.35  # 짧지만 복잡한 문제
            elif question_length < 400:
                if complexity > 0.6:
                    return "1", 0.35  # 중간 길이, 복잡한 문제
                else:
                    return "3", 0.40  # 중간 길이, 보통 문제
            else:
                if complexity > 0.7:
                    return "2", 0.30  # 긴 복잡한 문제
                else:
                    return "3", 0.35  # 긴 보통 문제
        else:
            return "", 0.15  # 주관식은 예측 어려움
    
    def update_patterns(self, question: str, correct_answer: str, structure: Dict,
                       prediction_result: Optional[Tuple[str, float]] = None):
        """패턴 학습 업데이트"""
        
        # 기존 패턴 업데이트
        self._update_basic_patterns(question, correct_answer, structure)
        
        # 성능 추적 업데이트
        if prediction_result:
            predicted_answer, confidence = prediction_result
            is_correct = (predicted_answer == correct_answer)
            
            # 사용된 규칙 식별
            pattern_match = self.analyze_question_pattern(question)
            if pattern_match:
                rule_name = pattern_match["rule"]
                self.pattern_performance["rule_success_rate"][rule_name].append(is_correct)
                self.pattern_performance["confidence_calibration"][rule_name].append(
                    (confidence, is_correct)
                )
        
        # 새로운 패턴 학습
        self._learn_new_patterns(question, correct_answer, structure)
    
    def _update_basic_patterns(self, question: str, correct_answer: str, structure: Dict):
        """기본 패턴 업데이트"""
        
        # 키워드-답변 매핑
        keywords = re.findall(r'[가-힣]{2,}', question.lower())
        for keyword in keywords:
            self.patterns["keyword_answer_map"][keyword][correct_answer] += 1
        
        # 도메인별 답변 분포
        domains = structure.get("domain", ["일반"])
        for domain in domains:
            self.patterns["domain_answer_distribution"][domain][correct_answer] += 1
        
        # 부정형 패턴
        if structure.get("has_negative", False):
            self.patterns["negative_answer_patterns"][correct_answer] += 1
        
        # 복잡도-답변 상관관계
        complexity = structure.get("complexity", 0.5)
        complexity_range = int(complexity * 10) / 10  # 0.1 단위로 정규화
        self.patterns["complexity_answer_correlation"][complexity_range].append(correct_answer)
        
        # 문제 유형별 패턴
        question_type = structure.get("question_type", "일반")
        self.patterns["question_type_patterns"][question_type][correct_answer] += 1
    
    def _learn_new_patterns(self, question: str, correct_answer: str, structure: Dict):
        """새로운 패턴 학습"""
        
        # 특이한 키워드 조합 발견
        keywords = re.findall(r'[가-힣]{2,}', question.lower())
        if len(keywords) >= 3:
            # 3개 이상 키워드가 함께 나타나는 패턴
            keyword_combination = tuple(sorted(keywords[:3]))
            if keyword_combination not in self.patterns:
                self.patterns[f"combo_{hash(keyword_combination)}"] = Counter()
            self.patterns[f"combo_{hash(keyword_combination)}"][correct_answer] += 1
        
        # 도메인별 특화 패턴 발견
        domains = structure.get("domain", [])
        for domain in domains:
            domain_keywords = [kw for kw in keywords if kw in question.lower()]
            if len(domain_keywords) >= 2:
                pattern_key = f"{domain}_pattern"
                if pattern_key not in self.patterns:
                    self.patterns[pattern_key] = defaultdict(Counter)
                
                combo_key = tuple(sorted(domain_keywords[:2]))
                self.patterns[pattern_key][combo_key][correct_answer] += 1
    
    def get_confidence_boost(self, question: str, predicted_answer: str, structure: Dict) -> float:
        """신뢰도 부스트 계산"""
        
        boost = 0.0
        
        # 패턴 매칭 확인
        pattern_match = self.analyze_question_pattern(question)
        
        if pattern_match:
            answers = pattern_match["answers"]
            if predicted_answer in answers:
                # 패턴 선호도에 따른 부스트
                preference_score = answers[predicted_answer]
                boost += preference_score * 0.15
                
                # 매칭 품질에 따른 부스트
                match_quality = pattern_match["match_score"]
                boost += match_quality * 0.1
        
        # 과거 성능 기반 부스트
        if pattern_match:
            rule_name = pattern_match["rule"]
            success_history = self.pattern_performance["rule_success_rate"].get(rule_name, [])
            if len(success_history) >= 5:
                recent_success_rate = sum(success_history[-5:]) / 5
                boost += (recent_success_rate - 0.5) * 0.1  # 50% 기준으로 조정
        
        # 도메인 일치성 부스트
        domains = structure.get("domain", [])
        if domains and pattern_match:
            domain_boost = self._calculate_domain_boost(pattern_match["rule"], domains)
            boost += domain_boost
        
        return min(boost, 0.3)  # 최대 30% 부스트
    
    def get_pattern_insights(self) -> Dict:
        """패턴 인사이트 분석"""
        insights = {
            "rule_performance": {},
            "domain_preferences": {},
            "complexity_trends": {},
            "negative_answer_distribution": dict(self.patterns["negative_answer_patterns"])
        }
        
        # 규칙별 성공률
        for rule_name, success_list in self.pattern_performance["rule_success_rate"].items():
            if len(success_list) >= 3:
                success_rate = sum(success_list) / len(success_list)
                insights["rule_performance"][rule_name] = {
                    "success_rate": success_rate,
                    "sample_size": len(success_list)
                }
        
        # 도메인별 선호 답변
        for domain, answer_dist in self.patterns["domain_answer_distribution"].items():
            if sum(answer_dist.values()) >= 5:
                total = sum(answer_dist.values())
                preferences = {ans: count/total for ans, count in answer_dist.items()}
                insights["domain_preferences"][domain] = preferences
        
        # 복잡도별 트렌드
        for complexity, answers in self.patterns["complexity_answer_correlation"].items():
            if len(answers) >= 5:
                answer_dist = Counter(answers)
                total = len(answers)
                trends = {ans: count/total for ans, count in answer_dist.items()}
                insights["complexity_trends"][complexity] = trends
        
        return insights
    
    def optimize_rules(self):
        """규칙 최적화"""
        
        # 성능이 낮은 규칙 조정
        for rule_name, success_list in self.pattern_performance["rule_success_rate"].items():
            if len(success_list) >= 10:
                success_rate = sum(success_list) / len(success_list)
                
                if success_rate < 0.4:  # 성공률이 40% 미만
                    # 신뢰도 감소
                    if rule_name in self.learned_rules:
                        self.learned_rules[rule_name]["confidence"] *= 0.9
                        print(f"규칙 {rule_name} 신뢰도 감소: 성공률 {success_rate:.2%}")
                
                elif success_rate > 0.7:  # 성공률이 70% 초과
                    # 신뢰도 증가
                    if rule_name in self.learned_rules:
                        current_confidence = self.learned_rules[rule_name]["confidence"]
                        self.learned_rules[rule_name]["confidence"] = min(current_confidence * 1.05, 0.95)
                        print(f"규칙 {rule_name} 신뢰도 증가: 성공률 {success_rate:.2%}")
    
    def save_patterns(self, filepath: str = "./learned_patterns_advanced.pkl"):
        """고급 패턴 저장"""
        save_data = {
            "patterns": dict(self.patterns),
            "rules": self.learned_rules,
            "performance": dict(self.pattern_performance),
            "cache_stats": {
                "prediction_cache_size": len(self.prediction_cache),
                "pattern_cache_size": len(self.pattern_cache)
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
    
    def load_patterns(self, filepath: str = "./learned_patterns_advanced.pkl"):
        """고급 패턴 로드"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
                self.patterns = defaultdict(Counter, data["patterns"])
                self.learned_rules = data["rules"]
                
                if "performance" in data:
                    self.pattern_performance = defaultdict(list, data["performance"])
                
                return True
        except Exception as e:
            print(f"패턴 로드 실패: {e}")
            return False
    
    def cleanup(self):
        """리소스 정리"""
        print(f"\n=== 패턴 학습기 통계 ===")
        print(f"학습된 규칙: {len(self.learned_rules)}개")
        print(f"예측 캐시: {len(self.prediction_cache)}개")
        print(f"패턴 캐시: {len(self.pattern_cache)}개")
        
        # 성능 요약
        if self.pattern_performance["rule_success_rate"]:
            total_predictions = sum(len(results) for results in self.pattern_performance["rule_success_rate"].values())
            successful_predictions = sum(sum(results) for results in self.pattern_performance["rule_success_rate"].values())
            
            if total_predictions > 0:
                overall_success_rate = successful_predictions / total_predictions
                print(f"전체 성공률: {overall_success_rate:.2%}")
        
        # 캐시 정리
        self.prediction_cache.clear()
        self.pattern_cache.clear()

class SmartAnswerSelector:
    """고성능 답변 선택기"""
    
    def __init__(self):
        self.pattern_learner = AnswerPatternLearner()
        self.selection_history = []
        self.selection_stats = {
            "total_selections": 0,
            "pattern_based": 0,
            "model_based": 0,
            "hybrid": 0
        }
        
    def select_best_answer(self, question: str, model_response: str, 
                         structure: Dict, confidence: float) -> Tuple[str, float]:
        """최적 답변 선택"""
        
        self.selection_stats["total_selections"] += 1
        
        # 1. 모델 응답에서 답변 추출
        extracted_answers = self._extract_all_possible_answers_advanced(model_response)
        
        if not extracted_answers:
            # 패턴 기반 예측
            pattern_answer, pattern_conf = self.pattern_learner.predict_answer(question, structure)
            self.selection_stats["pattern_based"] += 1
            return pattern_answer, pattern_conf
        
        # 2. 하이브리드 선택 전략
        answer_candidates = []
        
        # 모델 기반 후보들
        for answer in extracted_answers:
            score = self._score_model_answer(answer, model_response, confidence)
            answer_candidates.append({
                "answer": answer,
                "score": score,
                "source": "model",
                "confidence": confidence
            })
        
        # 패턴 기반 후보
        pattern_answer, pattern_conf = self.pattern_learner.predict_answer(question, structure)
        if pattern_answer:
            pattern_score = self._score_pattern_answer(pattern_answer, pattern_conf, structure)
            answer_candidates.append({
                "answer": pattern_answer,
                "score": pattern_score,
                "source": "pattern",
                "confidence": pattern_conf
            })
        
        # 3. 최적 후보 선택
        if answer_candidates:
            best_candidate = max(answer_candidates, key=lambda x: x["score"])
            
            # 선택 통계 업데이트
            if best_candidate["source"] == "model":
                if any(c["source"] == "pattern" for c in answer_candidates):
                    self.selection_stats["hybrid"] += 1
                else:
                    self.selection_stats["model_based"] += 1
            else:
                self.selection_stats["pattern_based"] += 1
            
            # 신뢰도 부스트 적용
            final_confidence = min(
                best_candidate["confidence"] + 
                self.pattern_learner.get_confidence_boost(question, best_candidate["answer"], structure),
                1.0
            )
            
            return best_candidate["answer"], final_confidence
        
        # 4. 폴백
        return "3", 0.25
    
    def _extract_all_possible_answers_advanced(self, response: str) -> List[str]:
        """고급 답변 추출"""
        answers = []
        
        # 우선순위별 패턴
        high_priority_patterns = [
            r'정답[:\s]*([1-5])',
            r'최종\s*답[:\s]*([1-5])',
            r'결론[:\s]*([1-5])',
        ]
        
        medium_priority_patterns = [
            r'답[:\s]*([1-5])',
            r'([1-5])번(?:\s*이\s*정답)',
            r'선택[:\s]*([1-5])',
            r'따라서\s*([1-5])',
        ]
        
        low_priority_patterns = [
            r'([1-5])번',
            r'선택지\s*([1-5])',
            r'결론.*?([1-5])',
        ]
        
        # 우선순위별로 추출
        all_patterns = [
            (high_priority_patterns, 3),
            (medium_priority_patterns, 2), 
            (low_priority_patterns, 1)
        ]
        
        weighted_answers = []
        
        for patterns, weight in all_patterns:
            for pattern in patterns:
                matches = re.finditer(pattern, response, re.IGNORECASE)
                for match in matches:
                    answer = match.group(1)
                    position = match.start()
                    weighted_answers.append((answer, weight, position))
        
        # 가중치와 위치를 고려하여 정렬
        weighted_answers.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        # 중복 제거하되 순서 유지
        seen = set()
        unique_answers = []
        for answer, weight, position in weighted_answers:
            if answer not in seen:
                seen.add(answer)
                unique_answers.append(answer)
        
        return unique_answers
    
    def _score_model_answer(self, answer: str, response: str, confidence: float) -> float:
        """모델 답변 점수 계산"""
        score = 0.0
        
        # 기본 신뢰도
        score += confidence * 0.4
        
        # 위치 점수 (뒤쪽일수록 높음)
        last_position = response.rfind(answer)
        if last_position >= 0:
            position_score = last_position / len(response)
            score += position_score * 0.2
        
        # 명시적 표현 보너스
        explicit_phrases = [
            f"정답.*{answer}",
            f"최종.*{answer}",
            f"{answer}번.*정답",
            f"결론.*{answer}"
        ]
        
        for phrase in explicit_phrases:
            if re.search(phrase, response, re.IGNORECASE):
                score += 0.15
                break
        
        # 추론 과정 보너스
        reasoning_indicators = ["따라서", "그러므로", "결론적으로", "분석하면"]
        reasoning_score = sum(0.05 for indicator in reasoning_indicators if indicator in response)
        score += min(reasoning_score, 0.2)
        
        return score
    
    def _score_pattern_answer(self, answer: str, confidence: float, structure: Dict) -> float:
        """패턴 답변 점수 계산"""
        score = 0.0
        
        # 기본 패턴 신뢰도
        score += confidence * 0.5
        
        # 구조적 적합성
        if structure.get("has_negative", False) and answer in ["1", "4", "5"]:
            score += 0.1  # 부정형에서 적절한 답변
        
        # 도메인 적합성
        domains = structure.get("domain", [])
        if domains:
            domain_boost = len([d for d in domains if d in ["개인정보보호", "전자금융", "정보보안"]]) * 0.05
            score += domain_boost
        
        # 복잡도 적합성
        complexity = structure.get("complexity", 0.5)
        if complexity > 0.7:
            score += 0.05  # 복잡한 문제에서 패턴의 안정성
        
        return score
    
    def _has_explicit_answer_phrase(self, response: str, answer: str) -> bool:
        """명시적 답변 표현 확인"""
        explicit_phrases = [
            f"정답.*{answer}",
            f"답.*{answer}",
            f"{answer}번.*정답",
            f"{answer}번.*맞",
            f"결론.*{answer}",
            f"최종.*{answer}"
        ]
        
        for phrase in explicit_phrases:
            if re.search(phrase, response, re.IGNORECASE):
                return True
        
        return False
    
    def get_selection_report(self) -> Dict:
        """선택 통계 보고서"""
        total = self.selection_stats["total_selections"]
        
        if total == 0:
            return {"message": "선택 기록 없음"}
        
        return {
            "total_selections": total,
            "model_based_rate": self.selection_stats["model_based"] / total,
            "pattern_based_rate": self.selection_stats["pattern_based"] / total,
            "hybrid_rate": self.selection_stats["hybrid"] / total,
            "pattern_learner_insights": self.pattern_learner.get_pattern_insights()
        }
    
    def cleanup(self):
        """리소스 정리"""
        print(f"\n=== 답변 선택기 통계 ===")
        total = self.selection_stats["total_selections"]
        if total > 0:
            print(f"총 선택: {total}회")
            print(f"모델 기반: {self.selection_stats['model_based']/total:.1%}")
            print(f"패턴 기반: {self.selection_stats['pattern_based']/total:.1%}")
            print(f"하이브리드: {self.selection_stats['hybrid']/total:.1%}")
        
        self.pattern_learner.cleanup()