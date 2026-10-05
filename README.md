# Transcrição de áudio para desktop

Aplicativo desktop para Linux e Windows que transcreve áudio com WhisperX,
identifica falantes e permite revisar e exportar a transcrição. A interface é
feita com PySide6; a lógica de transcrição permanece separada da interface.

O primeiro pacote de distribuição é uma pasta `onedir` para Linux. O executável
inclui as bibliotecas necessárias, mas **não inclui os modelos de IA**: o
WhisperX baixa seus modelos no primeiro uso e os mantém no cache local do
Hugging Face. O pacote WhisperX também contém seu pequeno modelo VAD de
17 MB, necessário para detectar trechos de fala.

## Requisitos

- Python 3.10–3.12 de 64 bits.
- Conexão com a internet no primeiro uso para baixar os modelos do WhisperX.
- Token do Hugging Face com acesso aos modelos necessários pelo pyannote para
  diarização.
- Para sugestões via Gemini, uma API key do Google AI Studio (opcional).

## Executar durante o desenvolvimento

Em Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

No Windows, crie/ative o ambiente virtual pelo terminal e instale as mesmas
dependências:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

## Credenciais

Informe o token do Hugging Face na janela do aplicativo para habilitar a
diarização. A chave é mantida apenas na memória durante a execução e não é
gravada no repositório nem no executável.

A sugestão de nomes pelo Gemini é opcional. Ao solicitá-la, o aplicativo avisa
que o texto da transcrição (limitado a 12.000 caracteres) será enviado à API do
Google usando a API key fornecida pelo próprio usuário. A chave também não é
armazenada.

## Gerar o pacote Linux

Execute em um ambiente Linux com as dependências de build instaladas:

```bash
python -m pip install -r requirements-build.txt
./build_linux.sh
```

O resultado fica em `dist/TranscricaoDesktop/`. Distribua a pasta inteira,
preservando sua estrutura; não distribua apenas o binário `TranscricaoDesktop`.
O pacote Linux deve ser compilado em Linux. Para Windows, será necessário
executar um build equivalente em Windows; o PyInstaller não é cross-compiler.

Para validar os imports de inferência no pacote:

```bash
dist/TranscricaoDesktop/TranscricaoDesktop --check-runtime
```

## Testes

```bash
python -m unittest discover -s tests -v
```
