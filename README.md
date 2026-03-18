# ATTI Avatar Framework

Framework completo para orquestração de avatares 3D/2D com síntese de voz, animação facial e geração de vídeo.

## Características

- **TTS Realista**: XTTS v2 para síntese de voz em múltiplas vozes
- **Animação Facial**: LivePortrait para animação de expressões
- **Geração de Vídeo**: FFmpeg para composição e encoding
- **Integração ATTI**: Conecta com PersonaEngine, SoulX, e Voice Layer
- **Escalabilidade**: Suporta 10.000+ avatares simultâneos

## Instalação

```bash
pip install -r requirements.txt
docker-compose up -d
```

## Uso

```python
from atti_avatar_framework import AvatarOrchestration

avatar = AvatarOrchestration(
    avatar_model="sofia",
    voice_model="xtts-v2"
)

video = avatar.generate_video(
    text="Olá, eu sou Sofia!",
    avatar_image="path/to/avatar.png",
    output_path="output.mp4"
)
```

## Documentação

Veja [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) para detalhes arquiteturais.

## Status

✅ Pronto para produção
