/* Composer */
Mailpile.Composer = {};
Mailpile.Composer.Drafts = {};
Mailpile.Composer.Crypto = {};
Mailpile.Composer.Recipients = {};
Mailpile.Composer.Tooltips = {};
Mailpile.Composer.Body = {};
Mailpile.Composer.Attachments = {};

/* Composer - Create new instance of composer */
Mailpile.Composer.init = function(mid) {

  // Reset tabindex for To: field
  $('#search-query').attr('tabindex', '-1');

  // Load Crypto States
  // FIXME: needs dynamic support for multi composers on a page
  Mailpile.Composer.Crypto.LoadStates(mid);

  // Initialize select2
  Mailpile.Composer.Recipients.AddressField('compose-to-' + mid);

  if ($('#compose-cc-' + mid).val()) {
    Mailpile.Composer.Recipients.AddressField('compose-cc-' + mid);
  }

  if ($('#compose-bcc-' + mid).val()) {
    Mailpile.Composer.Recipients.AddressField('compose-bcc-' + mid);
  }

  // Save Text Composing Objects (move to data model)
  Mailpile.Composer.Drafts[mid] = Mailpile.Composer.Model;

  // Show Crypto Tooltips
  Mailpile.Composer.Crypto.UpdateEncryptionState(mid, function() {
    Mailpile.Composer.Tooltips.Signature();
    Mailpile.Composer.Tooltips.Encryption();
    Mailpile.Composer.Tooltips.ContactDetails();
  });

  // Initialize Attachments
  // FIXME: needs to be bound to unique ID that can be destroyed
  Mailpile.Composer.Attachments.Uploader({
    browse_button: 'compose-attachment-pick-' + mid,
    container: 'compose-attachments-' + mid,
    mid: mid
  });

  // Body
  Mailpile.Composer.Body.Setup(mid);
};
