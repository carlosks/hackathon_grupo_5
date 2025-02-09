import smtplib
import ssl
import torch
import torchvision.transforms as transforms
from torchvision import models
import cv2
import os
from PIL import Image
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# 🚀 1️⃣ Configuração e Carregamento do Modelo
MODEL_PATH = "/Users/carlosks/Desktop/Deputado/reconhecimento_facial_webcam/mobilenetv2_faca_vs_nao_faca.pth"
LIMIAR_CONFIDENCIA = 0.70
FRAME_SAVE_PATH = "frames_detectados"
EMAIL_ENVIADO = False  # Flag para garantir que o e-mail só seja enviado uma vez

# Criar pasta para salvar os frames detectados
os.makedirs(FRAME_SAVE_PATH, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(weights=None)  
num_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_features, 2)  

# Carregar pesos treinados
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()  # Modo de inferência

# 🚀 2️⃣ Transformações da Imagem (Deve ser igual ao treinamento)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 🚀 3️⃣ Função para Enviar E-mail
def enviar_email(frame_path):
    """
    Envia um e-mail para a empresa de segurança informando a detecção de um objeto cortante.
    """

    global EMAIL_ENVIADO
    if EMAIL_ENVIADO:
        return  # Não envia outro e-mail se já foi enviado anteriormente

    EMAIL_ENVIADO = True  # Define que o e-mail já foi enviado para evitar spam

    # Configurações do servidor SMTP
    remetente_email = "cks7805@gmail.com"  # Substitua pelo seu e-mail do Gmail
    senha_email = "rrak vxch ofyz yrtv"  # ATENÇÃO: use senhas de aplicativo para evitar problemas de segurança
    destinatario_email = "cks7805@gmail.com"
    assunto = "ALERTA: Objeto Cortante Detectado"

    # Corpo do e-mail
    corpo_email = f"""
    Prezados,

    Um objeto cortante foi detectado no sistema de monitoramento.

    Frame identificado: {frame_path}

    Atenciosamente,
    Equipe de Segurança
    """

    # Configuração do e-mail
    mensagem = MIMEMultipart()
    mensagem["From"] = "DETECSUL"
    mensagem["To"] = "CKS"
    mensagem["Subject"] = assunto
    mensagem.attach(MIMEText(corpo_email, "plain"))

    # Anexar imagem do frame detectado
    with open(frame_path, "rb") as anexo:
        parte = MIMEBase("application", "octet-stream")
        parte.set_payload(anexo.read())
        encoders.encode_base64(parte)
        parte.add_header("Content-Disposition", f"attachment; filename={os.path.basename(frame_path)}")
        mensagem.attach(parte)

    # Enviar o e-mail via SMTP
    contexto = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=contexto) as servidor:
        servidor.login(remetente_email, senha_email)
        servidor.sendmail(remetente_email, destinatario_email, mensagem.as_string())

    print("E-mail de alerta enviado com sucesso!")

# 🚀 4️⃣ Função para Detecção de Facas no Vídeo
def detect_knives(video_source):
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("❌ Erro ao acessar o vídeo.")
        return

    frame_count = 0  

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Fim do vídeo ou erro ao capturar frame.")
                break

            frame_count += 1  

            # Converter o frame do OpenCV para PIL e aplicar transformações
            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            input_tensor = transform(image_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)  

                prob_faca = probabilities[0, 0].item()
                prob_nao_faca = probabilities[0, 1].item()

            # 🔍 Debug: Mostrar as probabilidades reais
            print(f"🔍 Frame {frame_count} - Probabilidades -> Faca: {prob_faca:.4f}, Não Faca: {prob_nao_faca:.4f}")

            # 🚀 Exibir Alerta Apenas se a Confiança para Faca for Alta e a Confiança para Não Faca for Baixa
            if prob_faca >= LIMIAR_CONFIDENCIA and prob_faca > prob_nao_faca:
                print(f"✅ Faca detectada no frame {frame_count}! Confiança: {prob_faca:.2f}")

                frame_copy = frame.copy()

                height, width, _ = frame_copy.shape
                text_position = (width // 6, height // 6)  
                font_scale = 2.0  
                thickness = 5  
                color = (0, 0, 255)  

                # Adicionar texto ao alerta
                cv2.putText(frame_copy, f"FACA DETECTADA (Frame {frame_count})", text_position, 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

                # 🔴 Adicionar borda vermelha ao frame para destaque
                cv2.rectangle(frame_copy, (20, 20), (width - 20, height - 20), (0, 0, 255), 10)

                # Exibir o frame detectado
                cv2.imshow("🔪 Faca Detectada!", frame_copy)

                # Salvar a imagem detectada para envio no e-mail
                frame_save_name = os.path.join(FRAME_SAVE_PATH, f"frame_{frame_count}.jpg")
                cv2.imwrite(frame_save_name, frame_copy)
                print(f"📸 Frame salvo: {frame_save_name}")

                # Enviar e-mail de alerta **apenas na primeira detecção**
                enviar_email(frame_save_name)

                # Manter o alerta por 1.5 segundos
                cv2.waitKey(1500)
            
            else:
                cv2.imshow("Filme - Análise em Tempo Real", frame)
                cv2.waitKey(50)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("🛑 Detecção interrompida pelo usuário.")

    cap.release()
    cv2.destroyAllWindows()


# 🚀 5️⃣ Executar a Detecção
if __name__ == "__main__":
    detect_knives("/Users/carlosks/Desktop/Deputado/reconhecimento_facial_webcam/Facas.mp4")