"""
================================================================================
viseme_sync.py — Módulo de Sincronização Labial (Visema Sync Engine)
================================================================================
Projeto  : Humanos Digitais — Pipeline de Mídia para Avatares
Versão   : 1.0.0
Autores  : Equipe de IA — Pipeline de Avatares

Descrição
---------
Este módulo implementa a extração de visemas (unidades visuais da fala) a partir
de um arquivo de áudio e gera uma sequência de animação labial frame a frame.

Conceitos Fundamentais
----------------------
* FONEMA   : Unidade sonora mínima de uma língua. Ex.: /p/, /a/, /s/
* VISEMA   : Representação visual de um fonema — formato da boca ao pronunciá-lo.
             Diferentes fonemas podem compartilhar o mesmo visema (ex: /p/, /b/ e /m/
             produzem o mesmo formato visual de boca fechada).
* LIP CURVE: Curva contínua (0.0–1.0) que descreve a abertura da boca em cada frame.

Fluxo de processamento
----------------------
  Áudio (.wav)
      │
      ▼
  Extração de segmentos de energia  ──▶  Mapeamento Fonema→Visema
      │
      ▼
  Lista de Visemas com timestamps
      │
      ▼
  Geração da Lip Curve (interpolação suave)
      │
      ▼
  Lista[float] — um valor por frame [0.0 .. 1.0]

Dependências
------------
  pip install numpy scipy librosa soundfile

Uso rápido
----------
  engine = VisemeSyncEngine(fps=30)
  visemas = engine.extract_visemes("fala.wav")
  curva   = engine.generate_lip_curve(visemas, total_frames=150)
================================================================================
"""

from __future__ import annotations

import io
import logging
import math
import struct
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Configuração de logging (mensagens em português)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("VisemeSyncEngine")


# ===========================================================================
# SEÇÃO 1 — MAPEAMENTOS FONEMA → VISEMA
# ===========================================================================

# Tabela de visemas para o Português Brasileiro.
# Cada chave é um CÓDIGO DE VISEMA (letra ou sigla) e o valor é a DESCRIÇÃO
# do formato da boca mais a intensidade de abertura padrão (0.0–1.0).
VISEMA_DESCRICAO: Dict[str, Dict] = {
    # ── Visema FECHADO ────────────────────────────────────────────────────
    "FECHADO": {
        "descricao": "Lábios completamente fechados (consoantes bilabiais: p, b, m)",
        "abertura_padrao": 0.0,
        "cor_debug": (200, 100, 100),   # vermelho escuro (BGR para OpenCV)
    },
    # ── Visema DENTES ─────────────────────────────────────────────────────
    "DENTES": {
        "descricao": "Dentes aparentes, lábio inferior sob dentes superiores (f, v)",
        "abertura_padrao": 0.1,
        "cor_debug": (220, 180, 120),
    },
    # ── Visema ESTREITO ───────────────────────────────────────────────────
    "ESTREITO": {
        "descricao": "Abertura estreita, língua próxima ao palato (s, z, t, d, n, l, r)",
        "abertura_padrao": 0.25,
        "cor_debug": (180, 220, 120),
    },
    # ── Visema MEDIO ──────────────────────────────────────────────────────
    "MEDIO": {
        "descricao": "Abertura média, boca semi-aberta (e, é, o, ó, k, g, x, j, nh, lh)",
        "abertura_padrao": 0.50,
        "cor_debug": (120, 220, 180),
    },
    # ── Visema ABERTO ─────────────────────────────────────────────────────
    "ABERTO": {
        "descricao": "Boca bem aberta, maxilar abaixado (a, à, á, â)",
        "abertura_padrao": 0.90,
        "cor_debug": (100, 200, 100),
    },
    # ── Visema ARREDONDADO ────────────────────────────────────────────────
    "ARREDONDADO": {
        "descricao": "Lábios projetados e arredondados (u, ú, o fechado, ã)",
        "abertura_padrao": 0.40,
        "cor_debug": (100, 150, 240),
    },
    # ── Visema SILENCIO ───────────────────────────────────────────────────
    "SILENCIO": {
        "descricao": "Sem fonação — boca relaxada em posição neutra",
        "abertura_padrao": 0.0,
        "cor_debug": (200, 200, 200),
    },
}

# ---------------------------------------------------------------------------
# Mapeamento FONEMA (IPA simplificado / ortográfico) → CÓDIGO DE VISEMA
# Cobre os fonemas mais comuns do Português Brasileiro.
# ---------------------------------------------------------------------------
FONEMA_PARA_VISEMA: Dict[str, str] = {
    # ── Bilabiais oclusivas e nasal ───────────────────────────────────────
    "p": "FECHADO", "b": "FECHADO", "m": "FECHADO",
    # ── Labiodentais ──────────────────────────────────────────────────────
    "f": "DENTES",  "v": "DENTES",
    # ── Alveolares, dentais, laterais e vibrantes ─────────────────────────
    "t": "ESTREITO", "d": "ESTREITO", "n": "ESTREITO",
    "s": "ESTREITO", "z": "ESTREITO", "l": "ESTREITO",
    "r": "ESTREITO", "rr": "ESTREITO",
    # ── Palatais e velares ────────────────────────────────────────────────
    "lh": "MEDIO",  "nh": "MEDIO",
    "x":  "MEDIO",  "j":  "MEDIO",
    "ch": "MEDIO",
    "k":  "MEDIO",  "g":  "MEDIO",
    "qu": "MEDIO",  "gu": "MEDIO",
    # ── Vogais abertas ────────────────────────────────────────────────────
    "a":  "ABERTO", "á":  "ABERTO",
    "à":  "ABERTO", "â":  "ABERTO",
    # ── Vogais médias ─────────────────────────────────────────────────────
    "e":  "MEDIO",  "é":  "MEDIO",
    "ê":  "MEDIO",  "o":  "MEDIO",
    "ó":  "MEDIO",
    # ── Vogais fechadas e arredondadas ────────────────────────────────────
    "i":  "ESTREITO", "í": "ESTREITO",
    "u":  "ARREDONDADO", "ú": "ARREDONDADO",
    "ã":  "ARREDONDADO", "õ": "ARREDONDADO",
    # ── Ditongos comuns ───────────────────────────────────────────────────
    "ai": "ABERTO",    "au": "ARREDONDADO",
    "ei": "MEDIO",     "eu": "MEDIO",
    "oi": "ARREDONDADO",
    # ── Silêncio / pausa ──────────────────────────────────────────────────
    " ":  "SILENCIO",  ".": "SILENCIO",
    ",":  "SILENCIO",  "!": "SILENCIO",
    "?":  "SILENCIO",
}

# ---------------------------------------------------------------------------
# Duração típica (em milissegundos) de cada tipo de visema no PT-BR.
# Usada como fallback quando não temos timestamps precisos de alinhamento.
# ---------------------------------------------------------------------------
DURACAO_TIPICA_MS: Dict[str, float] = {
    "FECHADO":      80,
    "DENTES":       90,
    "ESTREITO":     100,
    "MEDIO":        110,
    "ABERTO":       130,
    "ARREDONDADO":  120,
    "SILENCIO":     200,
}


# ===========================================================================
# SEÇÃO 2 — FUNÇÕES AUXILIARES DE ÁUDIO
# ===========================================================================

def carregar_audio_wav(caminho: str) -> Tuple[np.ndarray, int]:
    """
    Carrega um arquivo WAV e retorna o sinal de áudio e a taxa de amostragem.

    Tenta usar librosa para suporte amplo de formatos; faz fallback para
    scipy/wave se librosa não estiver disponível.

    Parâmetros
    ----------
    caminho : str
        Caminho para o arquivo .wav

    Retorno
    -------
    Tuple[np.ndarray, int]
        (sinal_float32_mono, sample_rate_hz)
    """
    if not Path(caminho).exists():
        raise FileNotFoundError(f"Arquivo de áudio não encontrado: '{caminho}'")

    # ── Tentativa 1: librosa (melhor suporte a formatos variados) ─────────
    try:
        import librosa
        sinal, sr = librosa.load(caminho, sr=None, mono=True)
        logger.info(f"Áudio carregado via librosa: {len(sinal)/sr:.2f}s @ {sr}Hz")
        return sinal.astype(np.float32), int(sr)
    except ImportError:
        logger.warning("librosa não encontrado; tentando scipy...")

    # ── Tentativa 2: scipy ────────────────────────────────────────────────
    try:
        from scipy.io import wavfile
        sr, data = wavfile.read(caminho)
        # Converte para mono se necessário
        if data.ndim > 1:
            data = data.mean(axis=1)
        # Normaliza para float32 [-1, 1]
        if data.dtype == np.int16:
            sinal = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            sinal = data.astype(np.float32) / 2_147_483_648.0
        else:
            sinal = data.astype(np.float32)
        logger.info(f"Áudio carregado via scipy: {len(sinal)/sr:.2f}s @ {sr}Hz")
        return sinal, int(sr)
    except ImportError:
        logger.warning("scipy não encontrado; tentando wave (stdlib)...")

    # ── Tentativa 3: wave (biblioteca padrão — apenas PCM linear) ─────────
    with wave.open(caminho, "rb") as wf:
        sr        = wf.getframerate()
        n_ch      = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames  = wf.getnframes()
        raw       = wf.readframes(n_frames)

    # Desempacota os bytes para array numpy
    fmt = {1: np.int8, 2: np.int16, 4: np.int32}.get(sampwidth, np.int16)
    data = np.frombuffer(raw, dtype=fmt)

    # Converte para mono
    if n_ch > 1:
        data = data.reshape(-1, n_ch).mean(axis=1)

    sinal = data.astype(np.float32) / float(np.iinfo(fmt).max)
    logger.info(f"Áudio carregado via wave (stdlib): {len(sinal)/sr:.2f}s @ {sr}Hz")
    return sinal, sr


def detectar_segmentos_ativos(
    sinal: np.ndarray,
    sr: int,
    limiar_energia: float = 0.02,
    janela_ms: float = 20.0,
    suavizacao: int = 3,
) -> List[Tuple[float, float]]:
    """
    Detecta segmentos de FALA ATIVA no sinal de áudio usando energia RMS.

    Divide o sinal em janelas de `janela_ms` milissegundos. Janelas com
    energia RMS acima de `limiar_energia` são marcadas como ativas (fala).
    Aplica suavização temporal para evitar fragmentação excessiva.

    Parâmetros
    ----------
    sinal           : np.ndarray  — sinal de áudio mono float32
    sr              : int         — taxa de amostragem (Hz)
    limiar_energia  : float       — energia RMS mínima para considerar fala (0–1)
    janela_ms       : float       — duração de cada janela de análise (ms)
    suavizacao      : int         — nº de janelas para suavização por média móvel

    Retorno
    -------
    List[Tuple[float, float]]
        Lista de (inicio_seg, fim_seg) em SEGUNDOS dos segmentos ativos.
    """
    amostras_por_janela = int(sr * janela_ms / 1000.0)
    n_janelas = len(sinal) // amostras_por_janela

    # Calcula energia RMS de cada janela
    energias = np.array([
        np.sqrt(np.mean(sinal[i * amostras_por_janela:(i + 1) * amostras_por_janela] ** 2))
        for i in range(n_janelas)
    ])

    # Suavização por média móvel — evita segmentos muito curtos (ruídos)
    if suavizacao > 1:
        kernel = np.ones(suavizacao) / suavizacao
        energias = np.convolve(energias, kernel, mode="same")

    # Máscara booleana: True onde há fala ativa
    ativa = energias > limiar_energia

    # Agrupa janelas contíguas ativas em segmentos
    segmentos: List[Tuple[float, float]] = []
    inicio = None
    for i, esta_ativo in enumerate(ativa):
        t = i * janela_ms / 1000.0  # tempo em segundos
        if esta_ativo and inicio is None:
            inicio = t
        elif not esta_ativo and inicio is not None:
            segmentos.append((inicio, t))
            inicio = None
    if inicio is not None:
        segmentos.append((inicio, n_janelas * janela_ms / 1000.0))

    logger.info(f"Detecção de fala: {len(segmentos)} segmentos ativos encontrados")
    return segmentos


def texto_para_fonemas_pt(texto: str) -> List[str]:
    """
    Converte texto em português para uma sequência aproximada de fonemas.

    Esta é uma implementação rule-based (sem modelo de ML) que cobre a grande
    maioria dos padrões ortográficos do português brasileiro.

    Para projetos em produção, recomenda-se usar o allosaurus ou g2p-ptbr.

    Parâmetros
    ----------
    texto : str — texto em português (maiúsculas ou minúsculas)

    Retorno
    -------
    List[str] — lista de fonemas/grafemas
    """
    texto = texto.lower().strip()
    fonemas: List[str] = []
    i = 0

    # Substitui dígrafos (combinações de 2 letras que representam 1 som)
    # antes de processar letra por letra
    DIGRAFOS = [
        ("lh", "lh"), ("nh", "nh"), ("ch", "ch"),
        ("ss", "s"),  ("rr", "rr"), ("qu", "qu"),
        ("gu", "gu"),
    ]
    for digr, repr_ in DIGRAFOS:
        texto = texto.replace(digr, f"_{repr_}_")

    # Tokenização simples: divide em tokens (grafemas e dígrafos)
    tokens: List[str] = []
    j = 0
    while j < len(texto):
        ch = texto[j]
        if ch == "_":
            # Lê até o próximo "_" para recuperar o dígrafo
            fim = texto.index("_", j + 1)
            tokens.append(texto[j + 1:fim])
            j = fim + 1
        else:
            tokens.append(ch)
            j += 1

    # Mapeia cada token para um fonema/visema-chave
    for tok in tokens:
        if tok in FONEMA_PARA_VISEMA:
            fonemas.append(tok)
        elif tok == " ":
            fonemas.append(" ")   # pausa entre palavras
        # Ignora caracteres sem mapeamento (números, pontuação especial etc.)

    return fonemas


# ===========================================================================
# SEÇÃO 3 — CLASSE PRINCIPAL
# ===========================================================================

class VisemeSyncEngine:
    """
    Engine de sincronização labial baseada em visemas.

    Responsabilidades
    -----------------
    1. Carregar e analisar arquivos de áudio WAV.
    2. Extrair visemas com timestamps baseados na energia do sinal.
    3. Gerar uma curva de abertura labial suave para uso em animação.

    Parâmetros
    ----------
    fps : int
        Frame rate alvo da animação (padrão: 30 fps).
    limiar_energia : float
        Sensibilidade da detecção de fala (0.0–1.0; padrão: 0.02).
    """

    def __init__(self, fps: int = 30, limiar_energia: float = 0.02):
        if fps <= 0:
            raise ValueError(f"fps deve ser positivo, recebido: {fps}")
        self.fps             = fps
        self.limiar_energia  = limiar_energia
        self._duracao_frame  = 1.0 / fps   # duração de cada frame em segundos

        logger.info(
            f"VisemeSyncEngine inicializado — fps={fps}, "
            f"limiar_energia={limiar_energia}"
        )

    # ------------------------------------------------------------------
    # MÉTODO PÚBLICO 1: extract_visemes
    # ------------------------------------------------------------------
    def extract_visemes(self, audio_path: str) -> List[Dict]:
        """
        Método principal: extrai a sequência de visemas de um arquivo de áudio.

        Fluxo interno
        -------------
        1. Carrega o arquivo WAV.
        2. Detecta segmentos de fala ativa (energy-based VAD).
        3. Para cada segmento, estima os fonemas dominantes e mapeia para visemas.
        4. Calcula o timestamp e o frame de início/fim de cada visema.
        5. Calcula a intensidade (normalizada pela energia do segmento).

        Parâmetros
        ----------
        audio_path : str — caminho para o arquivo WAV de entrada

        Retorno
        -------
        List[Dict] — cada dicionário contém:
            {
              "visema"      : str,    # código do visema (ex: "ABERTO")
              "descricao"   : str,    # descrição humana
              "start_sec"   : float,  # início em segundos
              "end_sec"     : float,  # fim em segundos
              "start_frame" : int,    # frame de início (no fps configurado)
              "end_frame"   : int,    # frame de fim
              "intensity"   : float,  # abertura normalizada [0.0–1.0]
            }
        """
        logger.info(f"Extraindo visemas de: '{audio_path}'")

        # ── Passo 1: Carrega o áudio ──────────────────────────────────
        sinal, sr = carregar_audio_wav(audio_path)
        duracao_total = len(sinal) / sr
        logger.info(f"  Duração do áudio: {duracao_total:.3f}s")

        # ── Passo 2: Detecta segmentos de fala ───────────────────────
        segmentos = detectar_segmentos_ativos(
            sinal, sr, limiar_energia=self.limiar_energia
        )

        if not segmentos:
            logger.warning("Nenhum segmento de fala detectado. Retornando visema de silêncio.")
            return [self._criar_visema("SILENCIO", 0.0, duracao_total)]

        # ── Passo 3: Subdivisão dos segmentos em visemas individuais ─
        # Cada segmento ativo é dividido em sub-janelas com duração típica
        # de um fonema (~80–130 ms), gerando uma sequência plausível de visemas.
        visemas: List[Dict] = []

        for inicio, fim in segmentos:
            duracao_seg = fim - inicio
            if duracao_seg < 0.04:       # ignora segmentos < 40ms (artefatos)
                continue

            # Calcula energia média do segmento (usada como intensidade)
            s0 = int(inicio * sr)
            s1 = int(fim * sr)
            energia_seg = float(
                np.sqrt(np.mean(sinal[s0:s1] ** 2))
            )
            intensidade = float(np.clip(energia_seg / 0.3, 0.1, 1.0))

            # Estima a sequência de visemas via análise espectral simplificada
            sequencia = self._estimar_sequencia_visemas(sinal[s0:s1], sr)

            # Distribui os visemas ao longo do segmento
            n_vis = len(sequencia)
            dur_por_vis = duracao_seg / n_vis

            for k, cod_vis in enumerate(sequencia):
                t_ini = inicio + k * dur_por_vis
                t_fim = t_ini + dur_por_vis
                visemas.append(
                    self._criar_visema(cod_vis, t_ini, t_fim, intensidade)
                )

        # ── Passo 4: Adiciona silêncio no início / fim se necessário ─
        visemas = self._inserir_silencio_fronteiras(
            visemas, duracao_total
        )

        logger.info(f"  Total de visemas extraídos: {len(visemas)}")
        return visemas

    # ------------------------------------------------------------------
    # MÉTODO PÚBLICO 2: generate_lip_curve
    # ------------------------------------------------------------------
    def generate_lip_curve(
        self,
        visemes: List[Dict],
        total_frames: int,
        suavizacao: bool = True,
    ) -> List[float]:
        """
        Gera uma curva de abertura da boca (0.0–1.0) para cada frame.

        Usa interpolação cúbica entre os visemas para criar transições
        naturais (sem saltos bruscos entre frames).

        Parâmetros
        ----------
        visemes      : List[Dict]  — saída de extract_visemes()
        total_frames : int         — número total de frames do vídeo
        suavizacao   : bool        — aplica filtro gaussiano para suavizar picos

        Retorno
        -------
        List[float]  — lista com `total_frames` valores em [0.0, 1.0]
        """
        if total_frames <= 0:
            raise ValueError("total_frames deve ser > 0")

        # ── Inicializa a curva com zeros (posição fechada) ────────────
        curva = np.zeros(total_frames, dtype=np.float32)

        # ── Preenche a curva com os valores de intensidade dos visemas ─
        for vis in visemes:
            f_ini = max(0, vis["start_frame"])
            f_fim = min(total_frames - 1, vis["end_frame"])
            if f_ini >= total_frames:
                continue

            abertura_alvo = (
                VISEMA_DESCRICAO[vis["visema"]]["abertura_padrao"]
                * vis["intensity"]
            )
            curva[f_ini:f_fim + 1] = abertura_alvo

        # ── Interpolação suave com scipy (se disponível) ──────────────
        try:
            from scipy.ndimage import gaussian_filter1d
            # sigma=2.5 suaviza ~5 frames (~170ms a 30fps)
            curva = gaussian_filter1d(curva, sigma=2.5)
        except ImportError:
            # Fallback: média móvel simples (janela de 5 frames)
            kernel_size = 5
            kernel = np.ones(kernel_size) / kernel_size
            curva = np.convolve(curva, kernel, mode="same")

        # ── Normaliza para garantir range [0.0, 1.0] ─────────────────
        max_val = curva.max()
        if max_val > 1.0:
            curva /= max_val

        # ── Garante que silêncios = 0 ────────────────────────────────
        curva = np.clip(curva, 0.0, 1.0)

        logger.info(
            f"Lip curve gerada: {total_frames} frames | "
            f"max={curva.max():.3f}, mean={curva.mean():.3f}"
        )
        return curva.tolist()

    # ------------------------------------------------------------------
    # MÉTODO EXTRA: to_blend_shapes
    # ------------------------------------------------------------------
    def to_blend_shapes(
        self,
        lip_curve: List[float],
    ) -> List[Dict[str, float]]:
        """
        Converte a lip curve em parâmetros de Blend Shapes prontos para
        motores 3D (Unreal Engine, Unity, Blender).

        Cada frame produz um dicionário com os seguintes blend shapes:
          - jaw_open       : abertura da mandíbula
          - mouth_stretch  : esticamento lateral da boca
          - mouth_round    : arredondamento (vogais 'u', 'o')
          - lip_lower_down : descida do lábio inferior

        Parâmetros
        ----------
        lip_curve : List[float] — saída de generate_lip_curve()

        Retorno
        -------
        List[Dict[str, float]] — um dict por frame
        """
        resultado = []
        for v in lip_curve:
            resultado.append({
                "jaw_open":       float(v),
                "mouth_stretch":  float(v * 0.4),
                "mouth_round":    float(max(0, v - 0.5) * 0.6),
                "lip_lower_down": float(v * 0.3),
            })
        return resultado

    # ------------------------------------------------------------------
    # MÉTODOS PRIVADOS (auxiliares internos)
    # ------------------------------------------------------------------

    def _criar_visema(
        self,
        codigo: str,
        start_sec: float,
        end_sec: float,
        intensidade: float = 1.0,
    ) -> Dict:
        """Cria um dicionário de visema padronizado."""
        codigo = codigo if codigo in VISEMA_DESCRICAO else "SILENCIO"
        return {
            "visema":       codigo,
            "descricao":    VISEMA_DESCRICAO[codigo]["descricao"],
            "start_sec":    round(start_sec, 4),
            "end_sec":      round(end_sec, 4),
            "start_frame":  int(start_sec * self.fps),
            "end_frame":    max(int(end_sec * self.fps) - 1, 0),
            "intensity":    round(float(np.clip(intensidade, 0.0, 1.0)), 4),
        }

    def _estimar_sequencia_visemas(
        self,
        segmento: np.ndarray,
        sr: int,
    ) -> List[str]:
        """
        Estima uma sequência plausível de visemas para um segmento de fala
        usando análise espectral simplificada (centróide espectral).

        O centróide espectral indica o "brilho" do som:
          - Baixo  (~200–500 Hz) → vogais abertas/arredondadas (ABERTO, ARREDONDADO)
          - Médio  (~500–2k Hz)  → vogais médias / consoantes sonoras (MEDIO)
          - Alto   (>2k Hz)      → fricativas, sibilantes (ESTREITO, DENTES)
          - Muito alto           → oclusivas surdas, plosivas (FECHADO)

        Esta é uma heurística robusta mesmo sem modelos de ML treinados.
        """
        # Número de sub-janelas baseado na duração típica de fonemas (100ms)
        dur_ms_por_visema = 100.0
        n_subdivisoes = max(1, int(len(segmento) / sr * 1000 / dur_ms_por_visema))

        sequencia: List[str] = []
        tamanho_sub = len(segmento) // n_subdivisoes

        for k in range(n_subdivisoes):
            sub = segmento[k * tamanho_sub:(k + 1) * tamanho_sub]
            if len(sub) == 0:
                continue

            # Aplica janela de Hann para reduzir vazamento espectral
            janela = np.hanning(len(sub))
            sub_janeado = sub * janela

            # Calcula o espectro de magnitude via FFT
            espectro = np.abs(np.fft.rfft(sub_janeado))
            freqs    = np.fft.rfftfreq(len(sub_janeado), d=1.0 / sr)

            # Centróide espectral: frequência média ponderada pela magnitude
            if espectro.sum() > 1e-10:
                centroide = float(np.sum(freqs * espectro) / np.sum(espectro))
            else:
                centroide = 0.0

            # Energia local (decide entre fala e silêncio)
            energia_local = float(np.sqrt(np.mean(sub ** 2)))

            # ── Regras de mapeamento centróide → visema ──────────────
            if energia_local < 0.01:
                visema_estimado = "SILENCIO"
            elif centroide < 400:
                visema_estimado = "ARREDONDADO"   # sons graves e arredondados
            elif centroide < 900:
                visema_estimado = "ABERTO"         # vogais abertas
            elif centroide < 1800:
                visema_estimado = "MEDIO"          # vogais médias
            elif centroide < 3500:
                visema_estimado = "ESTREITO"       # sibilantes / fricativas
            elif centroide < 5000:
                visema_estimado = "DENTES"         # labiodentais
            else:
                visema_estimado = "FECHADO"        # plosivas / oclusivas

            sequencia.append(visema_estimado)

        return sequencia if sequencia else ["SILENCIO"]

    def _inserir_silencio_fronteiras(
        self,
        visemas: List[Dict],
        duracao_total: float,
    ) -> List[Dict]:
        """
        Garante que há silêncio no início e no fim do vídeo,
        preenchendo lacunas antes do primeiro e depois do último visema.
        """
        resultado: List[Dict] = []

        # Silêncio antes do primeiro visema
        if visemas and visemas[0]["start_sec"] > 0.05:
            resultado.append(
                self._criar_visema("SILENCIO", 0.0, visemas[0]["start_sec"])
            )

        resultado.extend(visemas)

        # Silêncio após o último visema
        if visemas and visemas[-1]["end_sec"] < duracao_total - 0.05:
            resultado.append(
                self._criar_visema(
                    "SILENCIO",
                    visemas[-1]["end_sec"],
                    duracao_total,
                )
            )

        return resultado


# ===========================================================================
# SEÇÃO 4 — GERADOR DE ÁUDIO DE TESTE
# ===========================================================================

def gerar_audio_teste(
    caminho_saida: str = "./audio_teste_viseme.wav",
    duracao_s: float = 3.0,
    sr: int = 22050,
) -> str:
    """
    Gera um arquivo WAV de teste com padrão de fala sintético.

    O sinal mistura frequências vocálicas com envelope ADSR para simular
    a variação de energia típica da fala humana.

    Parâmetros
    ----------
    caminho_saida : str   — onde salvar o arquivo
    duracao_s     : float — duração em segundos
    sr            : int   — taxa de amostragem

    Retorno
    -------
    str — caminho do arquivo gerado
    """
    t = np.linspace(0, duracao_s, int(sr * duracao_s), endpoint=False)

    # Simula variação de energia típica da fala (padrão pulsado)
    sinal = np.zeros_like(t)
    # Adiciona "sílabas" a cada ~0.3s
    n_silabas = int(duracao_s / 0.3)
    for i in range(n_silabas):
        t0 = i * 0.3
        t1 = min(t0 + 0.2, duracao_s)
        mask = (t >= t0) & (t < t1)
        freq_fund = 150.0 + i * 20.0   # leve variação de pitch por sílaba
        for harm in [1, 2, 3]:
            sinal[mask] += (1.0 / harm) * np.sin(
                2 * np.pi * freq_fund * harm * t[mask]
            )
        # Envelope ADSR simplificado
        dur_amostras = mask.sum()
        if dur_amostras > 0:
            env = np.ones(dur_amostras)
            att = min(int(0.02 * sr), dur_amostras // 4)
            rel = min(int(0.05 * sr), dur_amostras // 4)
            env[:att]  = np.linspace(0, 1, att)
            env[-rel:] = np.linspace(1, 0, rel)
            sinal[mask] *= env

    # Normaliza
    max_v = np.max(np.abs(sinal))
    if max_v > 0:
        sinal = sinal / max_v * 0.85

    # Salva como WAV (PCM 16-bit)
    try:
        from scipy.io import wavfile
        wavfile.write(caminho_saida, sr, (sinal * 32767).astype(np.int16))
    except ImportError:
        # Fallback: escreve WAV manualmente (apenas 16-bit mono)
        samples = (sinal * 32767).astype(np.int16)
        with wave.open(caminho_saida, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(samples.tobytes())

    logger.info(f"Áudio de teste gerado: '{caminho_saida}' ({duracao_s}s, {sr}Hz)")
    return caminho_saida


# ===========================================================================
# SEÇÃO 5 — PONTO DE ENTRADA (EXEMPLO DE USO)
# ===========================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  VISEME SYNC ENGINE — DEMONSTRAÇÃO COMPLETA")
    print("=" * 65)

    # ── Passo 1: Gera um áudio de teste sintético ─────────────────────
    print("\n[1/4] Gerando áudio de teste sintético (3 segundos)...")
    audio_path = gerar_audio_teste("./audio_teste_viseme.wav", duracao_s=3.0)
    print(f"      Arquivo gerado: {audio_path}")

    # ── Passo 2: Instancia a engine ────────────────────────────────────
    print("\n[2/4] Instanciando VisemeSyncEngine (30 fps)...")
    engine = VisemeSyncEngine(fps=30, limiar_energia=0.02)

    # ── Passo 3: Extrai visemas ────────────────────────────────────────
    print("\n[3/4] Extraindo visemas do áudio...")
    visemas = engine.extract_visemes(audio_path)
    print(f"\n      Visemas extraídos ({len(visemas)} no total):")
    print(f"      {'Visema':<14} {'Início(s)':<12} {'Fim(s)':<10} "
          f"{'Frame ini':<12} {'Frame fim':<10} {'Intensidade'}")
    print("      " + "-" * 68)
    for v in visemas:
        print(
            f"      {v['visema']:<14} "
            f"{v['start_sec']:<12.3f} "
            f"{v['end_sec']:<10.3f} "
            f"{v['start_frame']:<12} "
            f"{v['end_frame']:<10} "
            f"{v['intensity']:.3f}"
        )

    # ── Passo 4: Gera a lip curve ──────────────────────────────────────
    print("\n[4/4] Gerando lip curve para 90 frames (3s × 30fps)...")
    total_frames = int(3.0 * 30)          # 3 segundos × 30 fps = 90 frames
    lip_curve    = engine.generate_lip_curve(visemas, total_frames=total_frames)

    print(f"\n      Primeiros 10 valores da curva labial:")
    print(f"      {[round(v, 3) for v in lip_curve[:10]]}")
    print(f"\n      Estatísticas da curva:")
    arr = np.array(lip_curve)
    print(f"        Mín   : {arr.min():.4f}")
    print(f"        Máx   : {arr.max():.4f}")
    print(f"        Média : {arr.mean():.4f}")
    print(f"        Desvio: {arr.std():.4f}")

    # ── Blend shapes ──────────────────────────────────────────────────
    blend_shapes = engine.to_blend_shapes(lip_curve)
    print(f"\n      Blend shapes (frame 0):  {blend_shapes[0]}")
    print(f"      Blend shapes (frame 30): {blend_shapes[min(30, len(blend_shapes)-1)]}")

    print("\n✅ Demonstração concluída com sucesso!")
    print(f"   Arquivo de áudio de teste: {audio_path}")
    print(f"   Curva labial: lista com {len(lip_curve)} valores float [0.0–1.0]")
    print("=" * 65)
