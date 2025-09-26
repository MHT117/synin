from django.test import TestCase
from django.urls import reverse

from .models import Conversation, Message


class ConversationSidebarTests(TestCase):
    def setUp(self):
        session = self.client.session
        session.save()
        self.session_key = session.session_key

    def test_list_conversations_filters_by_session(self):
        conv_mine = Conversation.objects.create(session_key=self.session_key, title="Mine")
        Message.objects.create(conversation=conv_mine, role="system", content="sys")
        Message.objects.create(conversation=conv_mine, role="user", content="hello")

        conv_other = Conversation.objects.create(session_key="other", title="Other")
        Message.objects.create(conversation=conv_other, role="system", content="sys")
        Message.objects.create(conversation=conv_other, role="user", content="hidden")

        resp = self.client.get(reverse('list_conversations'))
        self.assertEqual(resp.status_code, 200)
        data = resp.json().get('data', [])
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['id'], str(conv_mine.id))
        self.assertEqual(data[0]['message_count'], 2)

    def test_get_conversation_enforces_session(self):
        conv = Conversation.objects.create(session_key=self.session_key, title="Visible")
        Message.objects.create(conversation=conv, role="system", content="sys")
        Message.objects.create(conversation=conv, role="user", content="hi")

        other = Conversation.objects.create(session_key="other", title="Hidden")
        Message.objects.create(conversation=other, role="system", content="sys")

        ok = self.client.get(reverse('get_conversation', args=[conv.id]))
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(len(ok.json().get('messages', [])), 2)

        blocked = self.client.get(reverse('get_conversation', args=[other.id]))
        self.assertEqual(blocked.status_code, 404)
