from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from app.config import config
from app.models.schema import VideoParams


WEBUI_MAIN = Path(__file__).parent.parent.parent / "webui" / "Main.py"


def _widget(elements, key):
    return next(item for item in elements if str(item.key).startswith(key))


def _new_app():
    app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=30)
    app.session_state["ui_language"] = "en"
    app.run()
    assert not list(app.exception)
    return app


def _restore(app, params):
    app.session_state["task_restore_payload"] = {
        "task_id": "restored-scene-duration",
        "params": params,
    }
    app.run()
    assert not list(app.exception)
    return _widget(app.selectbox, "video_scene_min_duration_select")


def test_new_matching_task_uses_two_second_scene_minimum():
    with patch.object(config, "app", dict(config.app, match_materials_to_script=False)):
        app = _new_app()
        _widget(app.checkbox, "match_materials_to_script").set_value(True).run()

    minimum = _widget(app.selectbox, "video_scene_min_duration_select")
    assert minimum.value == 2.0
    assert minimum.disabled is False


def test_restored_legacy_task_keeps_scene_minimum_disabled():
    params = VideoParams(
        video_subject="legacy", match_materials_to_script=True
    ).model_dump(mode="json")
    params.pop("video_scene_min_duration")

    minimum = _restore(_new_app(), params)

    assert minimum.value is None


def test_restored_explicit_scene_minimum_is_preserved():
    params = VideoParams(
        video_subject="saved",
        match_materials_to_script=True,
        video_clip_duration=3,
        video_scene_min_duration=1.5,
    ).model_dump(mode="json")

    minimum = _restore(_new_app(), params)

    assert minimum.value == 1.5


def test_matching_toggle_preserves_selected_scene_minimum():
    app = _new_app()
    matching = _widget(app.checkbox, "match_materials_to_script")
    matching.set_value(True).run()
    _widget(app.selectbox, "video_scene_min_duration_select").set_value(1.5).run()

    _widget(app.checkbox, "match_materials_to_script").set_value(False).run()
    disabled = _widget(app.selectbox, "video_scene_min_duration_select")
    assert disabled.disabled is True
    assert disabled.value == 1.5

    _widget(app.checkbox, "match_materials_to_script").set_value(True).run()
    enabled = _widget(app.selectbox, "video_scene_min_duration_select")
    assert enabled.disabled is False
    assert enabled.value == 1.5
