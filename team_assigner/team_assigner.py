import cv2
from PIL import Image
from transformers import AutoModel, AutoProcessor

from utils.stub_utils import read_stub, save_stub


class TeamAssigner:

    def __init__(
        self, team1_class_name="Dark Blue Tshirt", team2_class_name="White Tshirt"
    ):
        self.team1_class_name = team1_class_name
        self.team2_class_name = team2_class_name

        self.player_dict = {}

    def load_model(self):
        self.model = AutoModel.from_pretrained(
            "patrickjohncyh/fashion-clip",
            attn_implementation="sdpa",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained("patrickjohncyh/fashion-clip")

    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]

        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)

        classes = [self.team1_class_name, self.team2_class_name]

        inputs = self.processor(
            text=classes,
            images=pil_img,
            return_tensors="pt",
            padding=True,
        )

        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

        class_name = classes[probs.argmax(dim=1)[0]]

        return class_name

    def get_player_team(self, frame, bbox, player_id):

        if player_id in self.player_dict.keys():
            return self.player_dict.get(player_id, "")

        player_color = self.get_player_color(frame, bbox)

        team_id = 1 if player_color == "Dark Blue Tshirt" else 2

        self.player_dict[player_id] = team_id

        return team_id

    def get_player_teams_across_frames(
        self, video_frames, player_tracks, read_from_stubs=False, stub_path=None
    ):

        player_assignments = read_stub(read_from_stubs, stub_path)

        if player_assignments is not None and len(player_assignments) == len(
            video_frames
        ):
            return player_assignments

        self.load_model()

        player_assignments = []

        for frame_num, player_track in enumerate(player_tracks):

            player_assignments.append({})

            if frame_num % 50 == 0:
                self.player_dict = {}

            for player_id, track in player_track.items():

                team = self.get_player_team(
                    video_frames[frame_num], track["bbox"], player_id
                )

                player_assignments[frame_num][player_id] = team

        save_stub(stub_path, player_assignments)

        return player_assignments
