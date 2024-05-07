/* Modals - Contacts */

Mailpile.UI.Modals.ContactAdd = function() {
  $('.sub-navigation ul li').removeClass('navigation-on');
  $(this).addClass('navigation-on');

  Mailpile.API.with_template('modal-contact-add', function(modal) {
    var modal_data = { name: '', address: '', extras: '' };
    $('#modal-full').html(modal(modal_data));
    $('#modal-full').modal(Mailpile.UI.ModalOptions);
  });
};


Mailpile.UI.Modals.ContactAddProcess = function() {
  Mailpile.API.contacts_add_post($('#form-contact-add').serialize(), function(result) {
    if (result.status == 'success') {
      $('#modal-full').modal('hide');

      // If Contacts List
      if (Mailpile.instance.state.command_url === "/contacts/") {
        var contact_template = _.template($('#template-contact-list-item').html());
        var contact_html = contact_template(result.result.contact);
        $('#contacts-list').append(contact_html);
      }
    }
  });
};
